"""
evaluate.py -- Phase 2: method-level evaluation of SBFL formulas
================================================================

Given a formula ``f(ep, ef, np, nf) -> float``, this module

  1. scores every statement with f (called once per *unique* spectrum),
  2. aggregates statement scores to METHODS with max aggregation
     (score of a method = highest score among its statements),
  3. computes the rank of the faulty method,
  4. reports acc@1 and wef (overall and per project).

Metrics
-------
* acc@1 : number of bugs whose faulty method is ranked FIRST.
* wef   : "wasted effort" = mean rank of the faulty method, i.e. how many
          methods a developer inspects (on average) before reaching the bug.
          Lower is better.

When a bug spans several methods, the best-ranked faulty method is used
(as allowed by the assignment).

Tie-breaking (IMPORTANT)
------------------------
Many methods often get exactly the same score.  Where does the faulty method
go inside a group of tied methods?  Let

    higher = number of NON-faulty methods with a strictly HIGHER score
    tied   = number of NON-faulty methods with EXACTLY the same score

Then ``CompiledBug.rank()`` returns three ranks:

    pessimistic (= conservative / worst case) : higher + tied + 1
        the faulty method is inspected LAST among the tied group
    average                                   : higher + tied / 2 + 1
        expected rank if the tied group is inspected in random order
    optimistic (= best case)                  : higher + 1
        the faulty method is inspected FIRST among the tied group

Example: 10 methods all tied at the top -> pessimistic 10, average 5.5,
optimistic 1.

What the code uses:
  * The GP fitness and the headline numbers (acc@1, wef) are PESSIMISTIC.
    With optimistic ties a useless formula that gives every statement the
    same score would tie everything and rank every bug first, so the GP would
    "cheat" (e.g. Wong1 = ef gets 103/119 optimistically but 2/119
    pessimistically).
  * Average-tie wef and optimistic acc@1 / wef are also reported, so results
    can be compared with work that uses another convention.
  * Note: for acc@1, average and pessimistic always agree -- the average rank
    is exactly 1 only when nothing is tied with (or above) the faulty method.

Implementation note
-------------------
The formula itself is always called with plain Python ints (never numpy
arrays), exactly as the grader will call best.sbfl().  numpy is used only
afterwards, to aggregate the resulting scores quickly.

Usage
-----
    python evaluate.py            # evaluate the baselines and best.py
"""
import math

import numpy as np

from data import load_spectra


def div(x, y):
    """Safe division required by the assignment: x / 0 is defined as 1."""
    if y == 0:
        return 1.0
    return x / y


# ---------------------------------------------------------------------------
# Pre-processing: compress each bug for fast repeated evaluation
# ---------------------------------------------------------------------------
class CompiledBug:
    """One bug, compressed for fast repeated evaluation by the GP.

    Observation: many statements share the same spectrum (ep, ef, np, nf),
    and a formula only depends on the spectrum.  So instead of calling the
    formula once per statement (~250k calls for all bugs), we call it once
    per UNIQUE spectrum (~20.5k calls for all bugs).

    Attributes:
        spectra     : list of unique (ep, ef, np, nf) tuples of this bug
        n_methods   : number of methods in the coverage data
        is_buggy    : bool array, is_buggy[m] = method m is a faulty method
        pair_spec   : flat array of unique-spectrum indices, grouped per method
        pair_starts : pair_spec[pair_starts[m] : pair_starts[m+1]] are the
                      unique spectra occurring in method m
    The pair_* layout lets np.maximum.reduceat compute every method's max
    score in a single vectorised call.
    """

    def __init__(self, bug):
        self.bug_id = bug["bug_id"]
        self.project = bug["project"]

        # Assign an index to every distinct spectrum, and remember which
        # spectra occur in which method.
        spec_index = {}                   # (ep, ef, np, nf) -> index
        method_specs = {}                 # method -> set of spectrum indices
        for method, ep, ef, np_, nf in bug["lines"]:
            s = (ep, ef, np_, nf)
            if s not in spec_index:
                spec_index[s] = len(spec_index)
            method_specs.setdefault(method, set()).add(spec_index[s])

        self.spectra = list(spec_index)               # unique (ep, ef, np, nf)
        methods = sorted(method_specs)
        self.n_methods = len(methods)
        self.is_buggy = np.array([m in bug["buggy_methods"] for m in methods])

        # Flatten "method -> spectra" into the reduceat layout (see docstring)
        pair_spec, starts = [], []
        for m in methods:
            starts.append(len(pair_spec))
            pair_spec.extend(sorted(method_specs[m]))
        self.pair_spec = np.array(pair_spec, dtype=np.int64)
        self.pair_starts = np.array(starts, dtype=np.int64)

    def method_scores(self, formula):
        """Return an array with the (max-aggregated) score of every method."""
        # 1) score each unique spectrum with plain Python ints
        scores = np.empty(len(self.spectra))
        for i, (ep, ef, np_, nf) in enumerate(self.spectra):
            try:
                v = float(formula(ep, ef, np_, nf))
            except (OverflowError, ZeroDivisionError, ValueError):
                v = float("nan")          # an invalid value counts as "lowest"
            scores[i] = v
        # NaN (e.g. from inf - inf) is treated as the lowest possible score
        scores[np.isnan(scores)] = -np.inf
        # 2) max aggregation: score of a method = max over its statements
        return np.maximum.reduceat(scores[self.pair_spec], self.pair_starts)

    def rank(self, formula):
        """Rank of the faulty method under the three tie-breaking schemes.

        Returns:
            (pessimistic, average, optimistic) -- see the module docstring.
        """
        ms = self.method_scores(formula)
        best = ms[self.is_buggy].max()      # best-scored faulty method
        others = ms[~self.is_buggy]         # all non-faulty methods
        higher = int((others > best).sum())   # strictly above the bug
        tied = int((others == best).sum())    # exactly tied with the bug
        pessimistic = higher + tied + 1       # bug inspected last in its tie
        average = higher + tied / 2 + 1       # random order inside the tie
        optimistic = higher + 1               # bug inspected first in its tie
        return pessimistic, average, optimistic


def compile_bugs(bugs=None):
    """Load spectra.pkl (unless bugs are given) and compress every bug."""
    if bugs is None:
        bugs = load_spectra()
    return [CompiledBug(b) for b in bugs]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
PROJECTS = ("Chart", "Lang", "Math", "Time")


def evaluate(formula, cbugs):
    """Evaluate one formula on a list of CompiledBug objects.

    Returns a dict with:
        acc@1, wef            : PESSIMISTIC ties (the headline numbers)
        wef_avg_tie           : average-tie wef
        acc@1_opt, wef_opt    : OPTIMISTIC ties
        n_bugs                : number of bugs evaluated
        ranks                 : {bug_id: (pessimistic, average, optimistic)}
        per_project           : the same metrics for Chart / Lang / Math / Time
    """
    ranks = {b.bug_id: b.rank(formula) for b in cbugs}
    pess = [ranks[b.bug_id][0] for b in cbugs]
    avg = [ranks[b.bug_id][1] for b in cbugs]
    opt = [ranks[b.bug_id][2] for b in cbugs]
    res = {
        "acc@1": sum(r == 1 for r in pess),
        "acc@1_opt": sum(r == 1 for r in opt),
        "wef_opt": float(np.mean(opt)),
        "n_bugs": len(cbugs),
        "wef": float(np.mean(pess)),
        "wef_avg_tie": float(np.mean(avg)),
        "ranks": ranks,
        "per_project": {},
    }
    for p in PROJECTS:
        sub = [b for b in cbugs if b.project == p]
        if not sub:
            continue
        pr = [ranks[b.bug_id][0] for b in sub]
        ar = [ranks[b.bug_id][1] for b in sub]
        orr = [ranks[b.bug_id][2] for b in sub]
        res["per_project"][p] = {
            "n": len(sub),
            "acc@1": sum(r == 1 for r in pr),
            "wef": float(np.mean(pr)),
            "wef_avg_tie": float(np.mean(ar)),
            "acc@1_opt": sum(r == 1 for r in orr),
            "wef_opt": float(np.mean(orr)),
        }
    return res


# ---------------------------------------------------------------------------
# Human-designed baselines (from the lecture / Yoo 2012)
#
# NOTE: these baselines use div0 (x/0 -> 0), the usual convention in the SBFL
# literature and in the course notebook.  With the assignment's safe div
# (x/0 -> 1), statements executed by NO test (ep = ef = 0) would get
# Ochiai = 1, the maximum possible score, which breaks the baseline.
# (Op2's denominator ep + np + 1 is never 0, so it can use div directly.)
# The evolved formula in best.py uses the assignment's div, as required.
# ---------------------------------------------------------------------------
def div0(x, y):
    """Division with x / 0 = 0 (literature convention, baselines only)."""
    return 0.0 if y == 0 else x / y


def ochiai(ep, ef, np, nf):
    """Ochiai = ef / sqrt((ef + nf) * (ef + ep))"""
    return div0(ef, math.sqrt((ef + nf) * (ef + ep)))


def tarantula(ep, ef, np, nf):
    """Tarantula = (ef/F) / (ef/F + ep/P)"""
    rf = div0(ef, ef + nf)
    rp = div0(ep, ep + np)
    return div0(rf, rf + rp)


def jaccard(ep, ef, np, nf):
    """Jaccard = ef / (ef + nf + ep)"""
    return div0(ef, ef + nf + ep)


def op2(ep, ef, np, nf):
    """Op2 (Naish et al.) = ef - ep / (ep + np + 1)"""
    return ef - div(ep, ep + np + 1)


def wong1(ep, ef, np, nf):
    """Wong1 = ef (ignores passing tests; produces many ties)"""
    return ef


def ample(ep, ef, np, nf):
    """AMPLE = | ef/F - ep/P |"""
    return abs(div0(ef, ef + nf) - div0(ep, ep + np))


BASELINES = {
    "Ochiai": ochiai,
    "Tarantula": tarantula,
    "Jaccard": jaccard,
    "Op2": op2,
    "Wong1": wong1,
    "AMPLE": ample,
}


def format_result(name, res):
    """One summary line: pessimistic acc@1/wef, plus optimistic acc@1 and
    average-tie wef, followed by per-project pessimistic numbers."""
    pp = "  ".join(
        f"{p}: {v['acc@1']:>2}/{v['n']:<2} wef={v['wef']:7.2f}"
        for p, v in res["per_project"].items()
    )
    return (f"{name:12s} acc@1={res['acc@1']:3d}/{res['n_bugs']} "
            f"(optimistic {res['acc@1_opt']:3d})  wef={res['wef']:6.2f} "
            f"(avg-tie {res['wef_avg_tie']:6.2f})  | {pp}")


if __name__ == "__main__":
    cbugs = compile_bugs()
    for name, f in BASELINES.items():
        print(format_result(name, evaluate(f, cbugs)))
    try:
        import best
        print(format_result("best.py", evaluate(best.sbfl, cbugs)))
    except Exception as e:  # best.py not filled in yet
        print("best.py:", e)
