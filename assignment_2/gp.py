"""
gp.py -- Phases 3-5: a self-written Genetic Programming (GP) engine that
evolves SBFL risk evaluation formulas (replicating Yoo, SSBSE 2012)
=========================================================================

No GP library is used (deap, pygad, ... are forbidden by the assignment);
every part of the algorithm below is implemented in this file.

Overview of one GP run
----------------------
    1. create a random population of formulas      (ramped half-and-half)
    2. repeat for `gens` generations:
         a. evaluate every formula on the training bugs   (fitness)
         b. keep the best few unchanged                     (elitism)
         c. fill the rest of the next population with children produced by
            crossover / mutation / reproduction of parents chosen by
            tournament selection
    3. return the best formula of the last generation

Representation
--------------
A formula is an expression TREE stored as nested tuples:

    ("div", "ef", ("sqrt", ("add", "ep", 1.0)))   ==   ef / sqrt(ep + 1)

    Terminals (leaves):        "ep", "ef", "np", "nf", and the constant 1.0
    Non-terminals (functions): add, sub, mul  (binary)
                               div   = the assignment's SAFE division (x/0 = 1)
                               sqrt  = protected square root, sqrt(|x|)

Tuples are immutable, so operators always build new trees and never modify a
parent by accident.

Fitness (lexicographic, MINIMISED; all ranks use PESSIMISTIC ties)
------------------------------------------------------------------
    --fitness acc  (default, used for the final formula)
                   key = (-acc@1, wef, size)
                   -> maximise acc@1; among equal acc@1 prefer lower wef;
                      among equal both prefer the smaller tree (anti-bloat)
    --fitness wef  key = (wef, -acc@1, size)
    --fitness log  key = (mean log2(rank), wef, size)

Pessimistic ties (see evaluate.py) make "tie everything" formulas useless,
so the GP cannot cheat with constant-like formulas.

Evaluation speed
----------------
Each tree is translated into Python source, compiled once into a lambda, and
called with one (ep, ef, np, nf) tuple of ints at a time -- no vector
arithmetic, exactly like best.sbfl().  Results are cached by source string,
because crossover often recreates formulas that were already evaluated.

Usage
-----
    # generalisation experiment: 60/40 train/validation split
    python gp.py --runs 6 --seed0 100 --pop 300 --gens 60 --out runs/acc_a.json
    # final runs: train on all 119 bugs
    python gp.py --train-all --runs 3 --seed0 400 --pop 300 --gens 60 --out runs/all_b.json
"""
import argparse
import json
import math
import random
import time

from data import load_spectra
from evaluate import CompiledBug, evaluate, div  # noqa: F401  (div used by eval)

# ---------------------------------------------------------------------------
# Primitive set
# ---------------------------------------------------------------------------
TERMINALS = ["ep", "ef", "np", "nf", 1.0]
FUNCTIONS = {"add": 2, "sub": 2, "mul": 2, "div": 2, "sqrt": 1}   # name -> arity

MAX_DEPTH = 8              # hard limit on tree depth (bloat control)
INIT_MIN, INIT_MAX = 2, 5  # depth range of the initial random trees


def psqrt(x):
    """Protected square root: sqrt(|x|), so negative inputs never crash."""
    return math.sqrt(abs(x))


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------
def is_terminal(t):
    """A leaf is anything that is not a tuple ("ep", "ef", ..., or 1.0)."""
    return not isinstance(t, tuple)


def to_python(t):
    """Translate a tree into a Python expression string.

    Example: ("div", "ef", ("sqrt", "ep")) -> "div(ef, psqrt(ep))"
    The string is used both to compile the formula and as the cache key.
    """
    if is_terminal(t):
        return repr(t) if isinstance(t, float) else t
    op = t[0]
    if op == "sqrt":
        return f"psqrt({to_python(t[1])})"
    a, b = to_python(t[1]), to_python(t[2])
    if op == "add":
        return f"({a} + {b})"
    if op == "sub":
        return f"({a} - {b})"
    if op == "mul":
        return f"({a} * {b})"
    return f"div({a}, {b})"                 # op == "div" (safe division)


def to_str(t):
    """Readable infix form, e.g. "(ef / sqrt(ep))" (used in logs/report)."""
    if is_terminal(t):
        return "1" if isinstance(t, float) else t
    op = t[0]
    if op == "sqrt":
        return f"sqrt({to_str(t[1])})"
    sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
    return f"({to_str(t[1])} {sym} {to_str(t[2])})"


def compile_tree(t):
    """Compile a tree into a callable f(ep, ef, np, nf) -> float.

    eval() only sees `div` and `psqrt`; the source comes from our own
    to_python(), never from user input.
    """
    src = to_python(t)
    return eval(f"lambda ep, ef, np, nf: {src}",
                {"div": div, "psqrt": psqrt})


def depth(t):
    """Depth of a tree (a single leaf has depth 0)."""
    if is_terminal(t):
        return 0
    return 1 + max(depth(c) for c in t[1:])


def size(t):
    """Number of nodes in a tree (used as an anti-bloat tie breaker)."""
    if is_terminal(t):
        return 1
    return 1 + sum(size(c) for c in t[1:])


def subtree_paths(t, path=()):
    """Yield the position of every node as a path of child indices.

    Example for ("add", "ep", ("sqrt", "ef")):
        ()      -> the whole tree
        (1,)    -> "ep"
        (2,)    -> ("sqrt", "ef")
        (2, 1)  -> "ef"
    """
    yield path
    if not is_terminal(t):
        for i, c in enumerate(t[1:], start=1):
            yield from subtree_paths(c, path + (i,))


def get_at(t, path):
    """Return the subtree located at `path`."""
    for i in path:
        t = t[i]
    return t


def replace_at(t, path, new):
    """Return a NEW tree where the subtree at `path` is replaced by `new`."""
    if not path:
        return new
    i = path[0]
    lst = list(t)
    lst[i] = replace_at(t[i], path[1:], new)
    return tuple(lst)


# ---------------------------------------------------------------------------
# Random tree generation
# ---------------------------------------------------------------------------
def random_tree(rng, max_depth, method="grow"):
    """Generate a random tree of depth <= max_depth.

    "full": every branch goes down to exactly max_depth (bushy trees)
    "grow": a branch may stop early (30% chance of a leaf at each level),
            which produces trees of varied shapes and sizes
    """
    if max_depth == 0 or (method == "grow" and rng.random() < 0.3):
        return rng.choice(TERMINALS)
    op = rng.choice(list(FUNCTIONS))
    kids = [random_tree(rng, max_depth - 1, method)
            for _ in range(FUNCTIONS[op])]
    return (op, *kids)


def ramped_half_and_half(rng, n):
    """Initial population: depths cycle through INIT_MIN..INIT_MAX ("ramped"),
    and half of the trees use "full", half use "grow" -> a diverse start."""
    pop = []
    depths = list(range(INIT_MIN, INIT_MAX + 1))
    for i in range(n):
        d = depths[i % len(depths)]
        pop.append(random_tree(rng, d, "full" if i % 2 else "grow"))
    return pop


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------
def crossover(rng, a, b):
    """Subtree crossover: replace a random subtree of `a` with a random
    subtree of `b`.  If the child would exceed MAX_DEPTH, return `a` unchanged.
    """
    pa = rng.choice(list(subtree_paths(a)))
    pb = rng.choice(list(subtree_paths(b)))
    child = replace_at(a, pa, get_at(b, pb))
    return child if depth(child) <= MAX_DEPTH else a


def subtree_mutation(rng, a):
    """Replace a random subtree of `a` with a NEW random tree of depth 1-3."""
    p = rng.choice(list(subtree_paths(a)))
    child = replace_at(a, p, random_tree(rng, rng.randint(1, 3)))
    return child if depth(child) <= MAX_DEPTH else a


def point_mutation(rng, a):
    """Change ONE node but keep the tree shape: a leaf becomes another leaf,
    a binary operator becomes another binary operator (sqrt stays sqrt)."""
    p = rng.choice(list(subtree_paths(a)))
    node = get_at(a, p)
    if is_terminal(node):
        new = rng.choice(TERMINALS)
    elif node[0] == "sqrt":
        new = node                    # only one unary op
    else:
        new = (rng.choice(["add", "sub", "mul", "div"]), node[1], node[2])
    return replace_at(a, p, new)


def tournament(rng, pop, fits, k):
    """Tournament selection: pick k random individuals, return the fittest
    (lowest fitness key).  Larger k = stronger selection pressure."""
    best = None
    for _ in range(k):
        i = rng.randrange(len(pop))
        if best is None or fits[i] < fits[best]:
            best = i
    return pop[best]


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------
class Fitness:
    """Fitness function with a cache keyed by the formula's source string.

    Calling an instance on a tree returns a tuple that is compared
    lexicographically (smaller = better); see the module docstring for the
    three variants.  Ranks are PESSIMISTIC (rank(f)[0], see evaluate.py).
    """

    def __init__(self, cbugs, kind="acc"):
        self.cbugs = cbugs     # training bugs (CompiledBug objects)
        self.kind = kind       # "acc" | "wef" | "log"
        self.cache = {}        # source string -> fitness key (without size)

    def __call__(self, t):
        key = to_python(t)
        if key not in self.cache:
            f = compile_tree(t)
            ranks = [b.rank(f)[0] for b in self.cbugs]   # [0] = pessimistic
            wef = sum(ranks) / len(ranks)
            acc = sum(r == 1 for r in ranks)
            if math.isnan(wef):
                wef = math.inf
            if self.kind == "wef":      # minimise wef, then maximise acc@1
                key2 = (wef, -acc)
            elif self.kind == "log":    # mean log2(rank): rewards top ranks,
                key2 = (sum(math.log2(r) for r in ranks) / len(ranks), wef)
            else:                       # "acc": maximise acc@1, then min wef
                key2 = (-acc, wef)
            self.cache[key] = key2
        # size is appended outside the cache: equal formulas, smaller tree wins
        return (*self.cache[key], size(t))


# ---------------------------------------------------------------------------
# One GP run
# ---------------------------------------------------------------------------
def run_gp(train, seed, pop_size=300, gens=60, tour_k=5, p_cx=0.8,
           p_mut=0.15, n_elite=5, kind="acc", verbose=True):
    """Run GP once and return (best_tree, history).

    Args:
        train    : training bugs (CompiledBug objects)
        seed     : random seed (the run is fully reproducible from it)
        pop_size : population size
        gens     : number of generations
        tour_k   : tournament size
        p_cx     : probability that a child is made by crossover
        p_mut    : probability that a child is made by mutation
                   (the remaining 1 - p_cx - p_mut is plain reproduction)
        n_elite  : best individuals copied unchanged to the next generation
        kind     : fitness variant ("acc", "wef" or "log")
    history = the first two fitness components of the best individual per
    generation (used for the convergence plot in the report).
    """
    rng = random.Random(seed)
    fitness = Fitness(train, kind)
    pop = ramped_half_and_half(rng, pop_size)
    history = []
    for g in range(gens + 1):
        # --- evaluate and sort the population (best first) ----------------
        fits = [fitness(t) for t in pop]
        order = sorted(range(len(pop)), key=lambda i: fits[i])
        best = pop[order[0]]
        history.append(fits[order[0]][:2])
        if verbose and (g % 10 == 0 or g == gens):
            f0 = fits[order[0]]
            print(f"  seed {seed} gen {g:3d}  fitness={f0[0]:8.3f},"
                  f"{f0[1]:8.3f}  size={f0[2]:3d}  {to_str(best)[:90]}",
                  flush=True)
        if g == gens:
            break
        # --- next generation: elitism + variation --------------------------
        nxt = [pop[i] for i in order[:n_elite]]           # elitism
        while len(nxt) < pop_size:
            r = rng.random()
            p1 = tournament(rng, pop, fits, tour_k)
            if r < p_cx:                                  # crossover (80%)
                child = crossover(rng, p1, tournament(rng, pop, fits, tour_k))
            elif r < p_cx + p_mut:                        # mutation (15%)
                child = (subtree_mutation(rng, p1) if rng.random() < 0.5
                         else point_mutation(rng, p1))
            else:                                         # reproduction (5%)
                child = p1
            nxt.append(child)
        pop = nxt
    return best, history


# ---------------------------------------------------------------------------
# Train / validation split (stratified by project, fixed seed)
# ---------------------------------------------------------------------------
def split(cbugs, train_frac=0.6, seed=454):
    """Split bugs into train/validation, keeping each project's proportion.

    With 119 bugs this gives 71 training and 48 validation bugs.  The fixed
    seed makes the split identical in every run, so runs are comparable.
    """
    rng = random.Random(seed)
    train, valid = [], []
    for p in sorted({b.project for b in cbugs}):
        group = [b for b in cbugs if b.project == p]
        rng.shuffle(group)
        k = round(len(group) * train_frac)
        train += group[:k]
        valid += group[k:]
    return train, valid


def main():
    """Command-line entry point: run several seeds and save results as JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--pop", type=int, default=300)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--out", default="runs.json")
    ap.add_argument("--fitness", default="acc", choices=["acc", "wef", "log"])
    ap.add_argument("--train-all", action="store_true",
                    help="train on all bugs (final run); validation = all")
    args = ap.parse_args()

    cbugs = [CompiledBug(b) for b in load_spectra()]
    train, valid = split(cbugs)
    if args.train_all:
        train = valid = cbugs
    print(f"{len(train)} training bugs, {len(valid)} validation bugs")

    results = []
    for seed in range(args.seed0, args.seed0 + args.runs):
        t0 = time.time()
        best, hist = run_gp(train, seed, args.pop, args.gens,
                            kind=args.fitness)
        # report the evolved formula on the train, validation and full sets
        f = compile_tree(best)
        tr, va, al = (evaluate(f, s) for s in (train, valid, cbugs))
        results.append({
            "seed": seed,
            "fitness": args.fitness,
            "formula": to_str(best),       # readable form
            "python": to_python(best),     # exact source used in best.py
            "tree": repr(best),            # tuple form (can be eval'd back)
            "size": size(best),
            "train": {"acc@1": tr["acc@1"], "wef": tr["wef"]},
            "valid": {"acc@1": va["acc@1"], "wef": va["wef"]},
            "all": {"acc@1": al["acc@1"], "wef": al["wef"]},
            "history": hist,
            "seconds": round(time.time() - t0, 1),
        })
        print(f"seed {seed}: valid acc@1={va['acc@1']} wef={va['wef']:.3f} | "
              f"all acc@1={al['acc@1']} wef={al['wef']:.3f}  "
              f"({results[-1]['seconds']}s)", flush=True)
        # save after every run, so an interruption loses at most one run
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
