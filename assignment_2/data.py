"""
data.py -- Phase 1: data preparation (raw coverage -> program spectra)
=====================================================================

What this module does
---------------------
For every bug in ``bug_data/`` it combines two input files:

* ``<bug>.json``      : which tests FAIL and which lines are BUGGY
* ``<bug>-cov.pkl``   : a coverage matrix (pandas DataFrame)
                        rows    = program statements ("lines")
                        columns = test cases
                        value   = True if the test executes the statement

and turns them into the *program spectrum* of every statement:

    ep = number of PASSING tests that execute the statement
    ef = number of FAILING tests that execute the statement
    np = number of PASSING tests that do NOT execute the statement
    nf = number of FAILING tests that do NOT execute the statement

Invariants (checked):  ep + np = P (all passing tests)
                       ef + nf = F (all failing tests)

Because the assignment ranks *methods* (not statements), every statement is
also tagged with the method it belongs to.

Why we cache
------------
The GP evaluates thousands of candidate formulas, each on every bug.  Loading
the raw pickles (some Time-* files are ~20 MB) every time would be far too
slow, so the spectra are computed ONCE and stored in ``spectra.pkl``.

Output format (one dict per bug, list saved in spectra.pkl)
-----------------------------------------------------------
    {
      "bug_id":        "Lang-12",
      "project":       "Lang",
      "n_fail":        2,                       # F
      "n_pass":        3,                       # P
      "lines":         [(method, ep, ef, np, nf), ...],   # one per statement
      "buggy_methods": {method, ...},           # ground truth (for scoring only)
      "buggy_in_cov":  {method, ...},           # buggy methods present in coverage
    }

All four counts are plain Python ``int`` (the assignment requires sbfl() to be
called with ints, never numpy values).

Usage
-----
    python data.py            # build spectra.pkl (reuses spectra_cache/ if present)
    python data.py --force    # recompute every bug from the raw files
"""
import os
import sys
import json
import pickle

import pandas as pd

DATA_DIR = "bug_data"          # raw input files provided with the assignment
CACHE = "spectra.pkl"          # final cache used by evaluate.py and gp.py
PER_BUG_DIR = "spectra_cache"  # per-bug cache so an interrupted build can resume


def method_of(line_id: str) -> str:
    """Return the method part of a statement identifier.

    Example:
        'org.apache.commons.lang3$StringUtils#join(java.lang.Object[],char):3'
        -> 'org.apache.commons.lang3$StringUtils#join(java.lang.Object[],char)'

    ``rsplit(":", 1)`` splits ONCE, from the RIGHT, so only the final
    ":<line number>" is removed even if the method signature itself were to
    contain other ':' characters.
    """
    return line_id.rsplit(":", 1)[0]


def load_bug(bug_id: str) -> dict:
    """Compute the spectra of every statement of one bug (see module docstring)."""
    # --- read the two raw input files -------------------------------------
    with open(os.path.join(DATA_DIR, f"{bug_id}.json")) as f:
        info = json.load(f)
    cov = pd.read_pickle(os.path.join(DATA_DIR, f"{bug_id}-cov.pkl"))

    # Sanity check: every failing test listed in the JSON must be a column of
    # the coverage matrix, otherwise ef/nf would silently be wrong.
    failing = set(info["failing_tests"])
    missing = failing - set(cov.columns)
    assert not missing, f"{bug_id}: failing tests not in coverage: {missing}"

    # --- coverage matrix as a dense boolean numpy array --------------------
    # IMPORTANT: the DataFrame uses a *sparse* boolean dtype.  On sparse
    # columns, DataFrame.sum(axis=1) behaves like a logical OR and returns
    # True/False instead of a COUNT (e.g. Lang-12 line 231 got ef = 1 instead
    # of 2).  Converting to a dense numpy matrix first avoids this bug.
    # X[i, j] == True  <=>  test j executes statement i
    X = cov.sparse.to_dense().to_numpy(dtype=bool) \
        if hasattr(cov, "sparse") else cov.to_numpy(dtype=bool)

    # Step 1: boolean mask over the columns -> True for failing tests
    fail_mask = cov.columns.isin(failing)
    n_fail = int(fail_mask.sum())          # F
    n_pass = len(fail_mask) - n_fail       # P

    # Step 2: count, per statement (row), how many failing / passing tests
    #         execute it (sum of True values across the selected columns)
    ef = X[:, fail_mask].sum(axis=1)
    ep = X[:, ~fail_mask].sum(axis=1)

    # Step 3: the "not executed" counts follow from the invariants
    #         ef + nf = F  and  ep + np = P
    nf = n_fail - ef
    np_ = n_pass - ep                      # "np_" so it does not shadow numpy
    assert (nf >= 0).all() and (np_ >= 0).all()

    # Step 4: one tuple per statement, converted to plain Python ints
    lines = [
        (method_of(line), int(a), int(b), int(c), int(d))
        for line, a, b, c, d in zip(cov.index, ep, ef, np_, nf)
    ]

    # Ground truth: the methods containing at least one buggy line.
    # (Used only to SCORE a formula, never as an input to the formula.)
    buggy_methods = {method_of(l) for l in info["buggy_lines"]}
    covered_methods = {m for m, *_ in lines}

    return {
        "bug_id": bug_id,
        "project": bug_id.split("-")[0],
        "n_fail": n_fail,
        "n_pass": n_pass,
        "lines": lines,                          # [(method, ep, ef, np, nf), ...]
        "buggy_methods": buggy_methods,
        # buggy methods that actually appear in the coverage data
        # (if empty, no formula could ever find the bug -- e.g. omission faults)
        "buggy_in_cov": buggy_methods & covered_methods,
    }


def build_all(force: bool = False) -> list:
    """Compute spectra for every bug and save them to CACHE.

    Each bug is also cached individually in PER_BUG_DIR, so an interrupted
    run can resume without recomputing the bugs that are already done
    (the Time-* coverage files are ~20MB each and slow to load).

    Args:
        force: if True, ignore PER_BUG_DIR and recompute every bug.
    """
    os.makedirs(PER_BUG_DIR, exist_ok=True)
    bug_ids = sorted(
        f[:-5] for f in os.listdir(DATA_DIR) if f.endswith(".json")
    )
    data = []
    for b in bug_ids:
        per_bug = os.path.join(PER_BUG_DIR, f"{b}.pkl")
        if os.path.exists(per_bug) and not force:
            with open(per_bug, "rb") as f:     # already computed: reuse
                bug = pickle.load(f)
        else:
            bug = load_bug(b)                  # compute from raw files
            with open(per_bug, "wb") as f:
                pickle.dump(bug, f)
        data.append(bug)
        # Progress line; warns if the buggy method is missing from coverage
        flag = "" if bug["buggy_in_cov"] else "  <-- buggy method NOT in coverage"
        print(f"{b:10s} lines={len(bug['lines']):6d} "
              f"fail={bug['n_fail']:2d} pass={bug['n_pass']:5d}{flag}", flush=True)
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved {len(data)} bugs to {CACHE}")
    return data


def load_spectra(path: str = CACHE) -> list:
    """Load the cached spectra (list of bug dicts). Used by evaluate.py and gp.py."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # use --force to recompute every bug instead of reusing spectra_cache/
    build_all(force="--force" in sys.argv)
