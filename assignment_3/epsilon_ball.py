"""Genetic algorithms that find epsilon-ball adversarial examples for VGG-11.

All three attacks share the same building blocks (see "Shared GA machinery"):
  * fitness   = log-probability margin  log p(true) - max_other log p(other)
                (< 0 means the model no longer predicts the true label)
  * evaluation of a whole generation in one batched forward pass
  * a wall-clock budget (TIME_BUDGET) so every call finishes well under 3 min
  * a final check with get_model_most_likely_for, exactly like the grader

Each norm then gets its own genotype and mutation operators:
  * L-inf : sign pattern s in {-1,+1}^(3x224x224); phenotype eps * s.
            Evolve s at a fixed eps, then lower eps whenever the GA succeeds.
  * L2    : real-valued noise. Start from an L-inf solution, then evolve
            "shrinking" mutations that keep the image adversarial while
            removing energy (util's L2 is a sum of squares).
  * L0    : a sparse set of (element index, value) genes with values at the
            clip boundary (+-3). Start dense, then evolve smaller sets.
"""
from util import *
import argparse
import math
import random
import time

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TIME_BUDGET = 150.0     # seconds per call (assignment limit: 180 s on CPU)
CHILDREN = 8            # offspring per generation = model batch size
ELITES = 3              # parents kept between generations ((mu + lambda) selection)
SUCCESS_MARGIN = -1e-3  # log-margin must be below this to count as adversarial
SEED = 454

H, W = IMG_SIZE
SHAPE = (1, 3, H, W)
N_ELEMS = 3 * H * W


# ---------------------------------------------------------------------------
# Shared GA machinery
# ---------------------------------------------------------------------------
def _seed_everything(seed=SEED):
    random.seed(seed)
    set_random_seed(seed)


class _Problem:
    """One attack run: original image, true label, fitness, time budget."""

    def __init__(self, image_path, budget=None):
        self.start = time.time()
        self.deadline = self.start + (TIME_BUDGET if budget is None else budget)
        self.x0 = get_image_tensor_from(image_path)
        _, pred = get_model_most_likely_for(self.x0)
        self.label = IMAGES_ORG_PREDS.get(image_path, pred)
        self.evals = 0

    def time_left(self):
        return self.deadline - time.time()

    def progress(self):
        total = self.deadline - self.start
        return min(1.0, max(0.0, (time.time() - self.start) / total))

    def apply(self, deltas):
        """Noise -> image. Values outside [-3, 3] are clipped by the model anyway."""
        return (self.x0 + deltas).clamp(-3, 3)

    def margins(self, deltas):
        """Fitness of a batch of noise tensors (lower is better, < 0 = success)."""
        imgs = self.apply(deltas)
        outs = []
        with torch.no_grad():
            for i in range(0, imgs.size(0), CHILDREN):
                outs.append(get_model_predictions_for(imgs[i:i + CHILDREN]))
        self.evals += imgs.size(0)
        logp = torch.cat(outs).clamp_min(1e-30).log()
        true = logp[:, self.label].clone()
        logp[:, self.label] = -float("inf")
        return true - logp.max(dim=1).values

    def is_adversarial(self, delta):
        """Same check as grade.py (single image, argmax != original label)."""
        _, idx = get_model_most_likely_for(self.apply(delta))
        return idx != self.label


def _square_size(progress, p_start=0.10, p_end=0.002):
    """Side of the square mutated: covers p_start of the image early, p_end late."""
    p = p_start * (p_end / p_start) ** progress
    return max(1, min(H, int(round(math.sqrt(p * H * W)))))


def _random_square(h):
    y = random.randint(0, H - h)
    x = random.randint(0, W - h)
    return y, x


def _block_mask(block):
    """Random 0/1 mask made of block x block tiles (for crossover)."""
    coarse = (torch.rand(1, 1, math.ceil(H / block), math.ceil(W / block)) < 0.5).float()
    return F.interpolate(coarse, scale_factor=block, mode="nearest")[:, :, :H, :W].bool()


def _crossover(a, b):
    """Uniform block crossover between two parent tensors."""
    mask = _block_mask(random.choice([8, 16, 32, 56]))
    return torch.where(mask, a, b)


def _fallback(P):
    """Last resort if the GA never succeeded: increasingly strong random noise."""
    for scale in [0.5, 1.0, 2.0, 4.0, 6.0]:
        delta = scale * torch.sign(torch.randn(SHAPE))
        if P.is_adversarial(delta):
            return delta
    return 6.0 * torch.sign(torch.randn(SHAPE))


def _first_verified(P, candidates, key):
    """Return the best candidate (by key) that passes the grader-style check."""
    for delta in sorted(candidates, key=key):
        if P.is_adversarial(delta):
            return delta
    return None


# ---------------------------------------------------------------------------
# L-inf: evolve sign patterns, shrink eps on success
# ---------------------------------------------------------------------------
def _linf_child(elites, h):
    a = random.choice(elites)[1]
    if len(elites) > 1 and random.random() < 0.5:
        b = random.choice(elites)[1]
        child = _crossover(a, b) if b is not a else a.clone()
    else:
        child = a.clone()
    # Mutation: give a random square a new sign per colour channel
    y, x = _random_square(h)
    child[0, :, y:y + h, x:x + h] = torch.sign(torch.randn(3, 1, 1))
    return child


def _calibrate_eps(P):
    """Try one stripe pattern at 8 noise levels (one batch); start the GA just
    below the smallest level that already fools the model."""
    s = torch.sign(torch.randn(1, 3, 1, W)).expand(SHAPE)
    levels = [0.02, 0.035, 0.05, 0.075, 0.11, 0.16, 0.24, 0.36]
    m = P.margins(torch.cat([e * s for e in levels])).tolist()
    for e, mi in zip(levels, m):
        if mi < SUCCESS_MARGIN:
            return 0.85 * e
    return levels[-1]


def _linf_search(P, eps=None, stop_at_first=False, until=None):
    """GA over sign patterns. Returns list of successful (eps, sign_pattern).
    Runs until the time budget is used up (or `until`, an absolute time)."""
    successes = []
    best_eps = float("inf")
    if eps is None:
        eps = _calibrate_eps(P)

    # Initial population: vertical stripes (one sign per channel/column) work
    # well as a starting point for L-inf attacks (Square Attack, Andriushchenko+ 2020)
    pop = [torch.sign(torch.randn(1, 3, 1, W)).expand(SHAPE).clone() for _ in range(CHILDREN)]
    m = P.margins(eps * torch.cat(pop))
    elites = sorted(zip(m.tolist(), pop), key=lambda e: e[0])[:ELITES]
    stagnation = 0
    gens_without_success = 0

    def reevaluate(new_eps):
        ss = [s for _, s in elites]
        mm = P.margins(new_eps * torch.cat(ss))
        return sorted(zip(mm.tolist(), ss), key=lambda e: e[0])

    until = P.deadline if until is None else until
    while time.time() < until:
        if elites[0][0] < SUCCESS_MARGIN:
            # Success at the current eps. Line search: how far can we shrink eps
            # with this same sign pattern? (one batch of CHILDREN candidates)
            s = elites[0][1]
            successes.append((eps, s))
            best_eps = min(best_eps, eps)
            if stop_at_first:
                break
            factors = torch.linspace(0.55, 0.97, CHILDREN)
            mm = P.margins(torch.cat([eps * f * s for f in factors]))
            ok = (mm < SUCCESS_MARGIN).nonzero().flatten()
            if len(ok) > 0:
                eps = eps * factors[ok[0]].item()
                successes.append((eps, s))
                best_eps = eps
            # Keep evolving just below the best eps found so far
            eps = 0.95 * best_eps
            elites = reevaluate(eps)
            stagnation = 0
            continue

        h = _square_size(P.progress())
        children = [_linf_child(elites, h) for _ in range(CHILDREN)]
        m = P.margins(eps * torch.cat(children))
        pool = elites + list(zip(m.tolist(), children))
        new_elites = sorted(pool, key=lambda e: e[0])[:ELITES]
        stagnation = 0 if new_elites[0][0] < elites[0][0] - 1e-4 else stagnation + 1
        elites = new_elites
        gens_without_success += 1

        if not successes and gens_without_success % 25 == 0:
            stagnation = 12  # slow progress counts as stuck while nothing works yet
        if stagnation >= 12:
            if not successes:
                eps *= 1.3                      # too hard: allow more noise
            else:
                eps = 0.5 * (eps + best_eps)    # step back towards last success
            elites = reevaluate(eps)
            stagnation = 0
    return successes


def make_Linf_adversarial_from(image_path):
    _seed_everything()
    P = _Problem(image_path)
    successes = _linf_search(P)
    deltas = [eps * s for eps, s in successes]
    delta = _first_verified(P, deltas, key=get_Linf_norm_for)
    if delta is None:
        delta = _fallback(P)
    return P.apply(delta)


# ---------------------------------------------------------------------------
# L2 (sum of squares): start adversarial, evolve towards less energy
# ---------------------------------------------------------------------------
def _effective(P, delta):
    """The noise that actually ends up in the image (after clipping)."""
    return P.apply(delta) - P.x0


L2_PHASE1_FRAC = 0.85   # share of the time budget spent on the L-inf search


def _l2_child(elites, h, strength):
    """Child noise from one or two adversarial parents."""
    a = random.choice(elites)[1]
    if len(elites) > 1 and random.random() < 0.3:
        b = random.choice(elites)[1]
        child = _crossover(a, b) if b is not a else a.clone()
    else:
        child = a.clone()

    r = random.random()
    if r < 0.55:
        # Local shrink: weaken (or remove) the noise in a random square
        y, x = _random_square(h)
        child[0, :, y:y + h, x:x + h] *= 1.0 - strength * random.random()
    elif r < 0.75:
        # Global shrink: scale everything down a little
        child *= 1.0 - 0.15 * strength * random.random()
    else:
        # Exploration: push energy into a random square, keep the total energy
        before = child.pow(2).sum()
        y, x = _random_square(h)
        amp = child.abs().max().item() * strength
        child[0, :, y:y + h, x:x + h] += amp * torch.sign(torch.randn(3, 1, 1))
        after = child.pow(2).sum()
        child *= torch.sqrt(before / after.clamp_min(1e-12)) * 0.99
    return child


def make_L2_adversarial_from(image_path):
    _seed_everything()
    P = _Problem(image_path)

    # Phase 1: L-inf GA. A small-eps sign pattern is already a good L2 solution
    # (L2 = eps^2 * #elements), much better than a random adversarial start.
    phase1_end = P.start + L2_PHASE1_FRAC * (P.deadline - P.start)
    successes = _linf_search(P, until=phase1_end)
    if not successes:
        successes = _linf_search(P, eps=0.2, stop_at_first=True)
    if not successes:
        return P.apply(_fallback(P))
    eps, s = min(successes, key=lambda e: e[0])
    delta = _effective(P, eps * s)
    elites = [(get_L2_norm_for(delta), delta)]
    found = [delta]

    # Phase 2: GA that only keeps adversarial children, ranked by L2.
    # Mutations remove energy where it does not seem to matter.
    strength = 0.8
    gen = 0
    while P.time_left() > 0:
        gen += 1
        h = _square_size(P.progress(), p_start=0.05, p_end=0.003)
        if gen % 10 == 0:
            # Every 10 generations: line search on a global scale factor
            best = elites[0][1]
            children = [best * f for f in torch.linspace(0.6, 0.98, CHILDREN)]
        else:
            children = [_l2_child(elites, h, strength) for _ in range(CHILDREN)]
        children = [_effective(P, c) for c in children]
        m = P.margins(torch.cat(children))
        ok = [(get_L2_norm_for(c), c) for mi, c in zip(m.tolist(), children) if mi < SUCCESS_MARGIN]

        # 1/5th success rule: adapt how aggressive the mutations are
        rate = len(ok) / len(children)
        strength = min(1.0, strength * 1.15) if rate > 0.2 else max(0.05, strength * 0.85)
        if ok:
            found.extend(c for _, c in ok)
            elites = sorted(elites + ok, key=lambda e: e[0])[:ELITES]

    found = sorted(found, key=get_L2_norm_for)[:20]
    delta = _first_verified(P, found, key=get_L2_norm_for)
    if delta is None:
        delta = _fallback(P)
    return P.apply(delta)


# ---------------------------------------------------------------------------
# L0: sparse genes (index, value), evolve fewer and fewer genes
# ---------------------------------------------------------------------------
def _l0_delta(P, idx, val):
    """Genes -> noise tensor: element idx is set to val (= +-3, the clip boundary)."""
    delta = torch.zeros(N_ELEMS)
    delta[idx] = val - P.x0.reshape(-1)[idx]
    return delta.view(SHAPE)


class _Sparse:
    """L0 individual: changed elements (idx), their values (val) and, per gene, a
    learned 'importance' score (how much the margin rose when that gene was removed)."""

    def __init__(self, idx, val, score=None, margin=None):
        self.idx, self.val = idx, val
        self.score = torch.zeros(len(idx)) if score is None else score
        self.margin = margin

    def __len__(self):
        return len(self.idx)

    def subset(self, keep):
        return _Sparse(self.idx[keep], self.val[keep], self.score[keep].clone())


def _random_sparse(k):
    idx = torch.randperm(N_ELEMS)[:k]
    val = 3.0 * torch.sign(torch.randn(k))
    return _Sparse(idx, val)


def _l0_child(parent, remove_frac):
    """Mutation: drop genes, preferring ones that looked unimportant, and sometimes
    move one gene to a new random element. Returns (child, removed positions)."""
    k = len(parent)
    n_remove = min(k - 1, max(1, int(round(remove_frac * k))))
    noise = torch.randn(k) * (parent.score.std().item() + 1e-3)
    order = torch.argsort(parent.score + noise)       # low score = unimportant
    removed, keep = order[:n_remove], order[n_remove:].sort().values
    child = parent.subset(keep)
    if random.random() < 0.2 and len(child) > 0:
        j = random.randrange(len(child))
        new_idx = random.randrange(N_ELEMS)
        if new_idx not in set(child.idx.tolist()):
            child.idx[j] = new_idx
            child.val[j] = 3.0 * random.choice([-1.0, 1.0])
            child.score[j] = 0.0
    return child, removed


def make_L0_adversarial_from(image_path):
    _seed_everything()
    P = _Problem(image_path)

    # Phase 1: find a (dense) random set of genes that works, doubling k when needed
    k = 1000
    elites = []
    while not elites and P.time_left() > 0:
        pop = [_random_sparse(k) for _ in range(CHILDREN)]
        m = P.margins(torch.cat([_l0_delta(P, g.idx, g.val) for g in pop])).tolist()
        for g, mi in zip(pop, m):
            g.margin = mi
        elites = [g for g in pop if g.margin < SUCCESS_MARGIN]
        k = min(N_ELEMS, k * 2)
    if not elites:
        return P.apply(_fallback(P))
    elites = sorted(elites, key=lambda g: (len(g), g.margin))[:ELITES]
    found = list(elites)

    # Phase 2: GA keeping only adversarial children, ranked by number of genes
    remove_frac = 0.2
    while P.time_left() > 0:
        parents = [random.choice(elites) for _ in range(CHILDREN)]
        children = [_l0_child(p, remove_frac) for p in parents]
        m = P.margins(torch.cat([_l0_delta(P, c.idx, c.val) for c, _ in children])).tolist()

        successes = 0
        for p, (c, removed), mi in zip(parents, children, m):
            c.margin = mi
            # Credit assignment: removing important genes raises the margin
            p.score[removed] += (mi - p.margin) / len(removed)
            if mi < SUCCESS_MARGIN:
                successes += 1
                found.append(c)
                elites.append(c)
        elites = sorted(elites, key=lambda g: (len(g), g.margin))[:ELITES]
        remove_frac = min(0.5, remove_frac * 1.25) if successes else max(1e-4, remove_frac * 0.7)

    found = sorted(found, key=len)[:20]
    deltas = [_l0_delta(P, g.idx, g.val) for g in found]
    delta = _first_verified(P, deltas, key=get_L0_norm_for)
    if delta is None:
        delta = _fallback(P)
    return P.apply(delta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='L-norm Adversarial Evolver')
    parser.add_argument('-f', '--filename', required=True, default='images/knot.jpg')
    parser.add_argument('-n', '--norm', type=str, required=True, default='inf')
    args = parser.parse_args()

    if args.norm.lower() == 'inf':
        adv_func = make_Linf_adversarial_from
    elif args.norm == '2':
        adv_func = make_L2_adversarial_from
    elif args.norm == '0':
        adv_func = make_L0_adversarial_from
    else:
        raise ValueError(f'Unknown norm type {args.norm}')

    adv_image = adv_func(args.filename)
    get_viz_for(adv_image)
