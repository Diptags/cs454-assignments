import argparse
import math
import random
import time
from datetime import datetime

import torch
from perlin_noise import PerlinNoise

from util import *

def perlin_image_generator(octaves):
    size = min(octaves*5, 224)
    noise = PerlinNoise(octaves=octaves, seed=1)
    pic = [[noise([i/size, j/size]) for j in range(size)] for i in range(size)]
    pic_tensor = TRANSFORM(torch.Tensor(pic).unsqueeze(0).repeat(3, 1, 1)).unsqueeze(0)
    pic_tensor = (pic_tensor - pic_tensor.min())/(pic_tensor.max()-pic_tensor.min())
    return pic_tensor

def genotype2phenotype(genes):
    noise = get_random_noise()*0
    for octave, magnitude in genes:
        noise += magnitude * perlin_image_generator(octave)
    return noise

def add_noise_to_image(noise, image):
    noise = noise.clip(0, 1)
    actual_white = torch.Tensor([[[2.2489]], [[2.428]], [[2.64]]]).repeat(1, 224, 224).unsqueeze(0)
    return (1-noise)*image + noise*actual_white

# ---------------------------------------------------------------------------
# Genetic algorithm over Perlin "fog" genotypes [(octave, magnitude), ...]
# ---------------------------------------------------------------------------
TIME_BUDGET = 270.0     # seconds per call (assignment limit: 300 s on CPU)
CHILDREN = 8            # offspring per generation = model batch size
ELITES = 3
SUCCESS_MARGIN = -1e-3  # log-prob margin must go below this to count as adversarial
MAX_OCTAVE = 40         # octave search range 1..MAX_OCTAVE (pattern cost grows with octave)
MAX_GENES = 6           # keep genotypes short (assignment allows up to 50)
SEED = 454

_PATTERNS = {}


def _pattern(octave):
    """perlin_image_generator is deterministic (seed=1), so compute each octave once."""
    if octave not in _PATTERNS:
        _PATTERNS[octave] = perlin_image_generator(octave)
    return _PATTERNS[octave]


def _phenotype(genes):
    """Same arithmetic as genotype2phenotype, but with cached patterns."""
    noise = get_random_noise() * 0
    for octave, magnitude in genes:
        noise += magnitude * _pattern(octave)
    return noise


def _normalise(genes):
    """Merge duplicate octaves, drop tiny genes, keep at most MAX_GENES."""
    merged = {}
    for octave, magnitude in genes:
        octave = int(min(MAX_OCTAVE, max(1, octave)))
        merged[octave] = merged.get(octave, 0.0) + magnitude
    out = sorted((o, m) for o, m in merged.items() if m > 1e-4)
    out = sorted(out, key=lambda g: -g[1])[:MAX_GENES]
    return sorted(out) if out else [(random.randint(1, 8), 0.1)]


def _strength(genes):
    """The assignment's epsilon for fog: sum of magnitudes."""
    return sum(m for _, m in genes)


def _scale(genes, a):
    return [(o, m * a) for o, m in genes]


class _FogProblem:
    def __init__(self, image_path):
        self.start = time.time()
        self.deadline = self.start + TIME_BUDGET
        self.x0 = get_image_tensor_from(image_path)
        _, pred = get_model_most_likely_for(self.x0)
        self.label = IMAGES_ORG_PREDS.get(image_path, pred)
        self.m0 = self.margins([[]])[0]

    def time_left(self):
        return self.deadline - time.time()

    def margins(self, genotypes):
        imgs = torch.cat([add_noise_to_image(_phenotype(g), self.x0) for g in genotypes])
        with torch.no_grad():
            probs = get_model_predictions_for(imgs)
        logp = probs.clamp_min(1e-30).log()
        true = logp[:, self.label].clone()
        logp[:, self.label] = -float("inf")
        return (true - logp.max(dim=1).values).tolist()

    def boundary_scale(self, m):
        """Fog is linear in the magnitudes (before clipping), and the margin is a logit
        difference, so margin(a * genes) ~ m0 + a * (m - m0). Solve for the boundary."""
        drop = self.m0 - m
        return (self.m0 - SUCCESS_MARGIN) / drop if drop > 1e-6 else float("inf")

    def is_adversarial(self, genes, official=False):
        """Grader-style check (single image, argmax). official=True regenerates the
        fog from scratch with genotype2phenotype, exactly like the grader (slower)."""
        noise = genotype2phenotype(genes) if official else _phenotype(genes)
        _, idx = get_model_most_likely_for(add_noise_to_image(noise, self.x0))
        return idx != self.label


def _random_genotype():
    n = random.randint(1, 3)
    return _normalise([(random.randint(1, 16), random.uniform(0.2, 1.0)) for _ in range(n)])


def _mutate(genes):
    genes = list(genes)
    r = random.random()
    if r < 0.35:
        # Change one magnitude (log-normal step)
        j = random.randrange(len(genes))
        o, m = genes[j]
        genes[j] = (o, m * math.exp(random.gauss(0, 0.4)))
    elif r < 0.6:
        # Nudge one octave
        j = random.randrange(len(genes))
        o, m = genes[j]
        genes[j] = (o + random.choice([-3, -2, -1, 1, 2, 3]), m)
    elif r < 0.75:
        # Replace one octave with a random one
        j = random.randrange(len(genes))
        genes[j] = (random.randint(1, MAX_OCTAVE), genes[j][1])
    elif r < 0.9:
        # Add a gene
        total = _strength(genes)
        genes.append((random.randint(1, MAX_OCTAVE), total * random.uniform(0.05, 0.4)))
    elif len(genes) > 1:
        # Remove a gene
        genes.pop(random.randrange(len(genes)))
    return _normalise(genes)


def _crossover(a, b):
    """Each octave of either parent is inherited with probability 1/2."""
    pool = list(a) + list(b)
    child = [g for g in pool if random.random() < 0.5]
    return _normalise(child or [random.choice(pool)])


def adv_func(image_path):
    random.seed(SEED)
    set_random_seed(SEED)
    P = _FogProblem(image_path)

    def rescaled(genes, m):
        """Fitness: total magnitude after moving the fog onto the decision boundary."""
        a = P.boundary_scale(m)
        if not math.isfinite(a):
            return float("inf"), genes
        genes = _scale(genes, a)
        return _strength(genes), genes

    # Initial population: random short genotypes
    pop = [_random_genotype() for _ in range(CHILDREN)]
    mm = P.margins(pop)
    found = [g for g, m in zip(pop, mm) if m < SUCCESS_MARGIN]
    elites = sorted([rescaled(g, m) for g, m in zip(pop, mm)], key=lambda e: e[0])[:ELITES]

    gen = 0
    while P.time_left() > 0:
        gen += 1
        if gen % 4 == 0:
            # Line search on the best genotype: find where it really crosses the boundary
            best = elites[0][1]
            factors = [0.8, 0.86, 0.92, 0.97, 1.0, 1.05, 1.12, 1.22]
            cands = [_scale(best, f) for f in factors]
            mm = P.margins(cands)
            ok = [c for c, m in zip(cands, mm) if m < SUCCESS_MARGIN]
            found.extend(ok)
            if ok:
                elites[0] = (_strength(ok[0]), ok[0])
            else:
                s, g = rescaled(cands[-1], mm[-1])
                elites[0] = (s * 1.1, g)
            elites = sorted(elites, key=lambda e: e[0])
            continue

        children = []
        for _ in range(CHILDREN):
            a = random.choice(elites)[1]
            if len(elites) > 1 and random.random() < 0.3:
                a = _crossover(a, random.choice(elites)[1])
            children.append(_mutate(a))
        mm = P.margins(children)
        found.extend(g for g, m in zip(children, mm) if m < SUCCESS_MARGIN)
        scored = [rescaled(g, m) for g, m in zip(children, mm)]
        elites = sorted(elites + scored, key=lambda e: e[0])[:ELITES]

    # Final answer: weakest fog that passes the grader-style check
    for genes in sorted(found, key=_strength)[:20]:
        if P.is_adversarial(genes) and P.is_adversarial(genes, official=True):
            return [(int(o), float(m)) for o, m in genes]
    # Fallback: thick fog
    for a in [1.0, 2.0, 4.0]:
        genes = [(1, a), (8, a)]
        if P.is_adversarial(genes):
            return genes
    return [(1, 8.0)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Perlin Adversarial Evolver')
    parser.add_argument('-f', '--filename', required=True, default='images/knot.jpg')
    args = parser.parse_args()
    
    noise_genotype = adv_func(args.filename)
    noise_img = genotype2phenotype(noise_genotype)
    org_img = get_image_tensor_from(args.filename)
    adv_image = add_noise_to_image(noise_img, org_img)
    get_viz_for(adv_image)
