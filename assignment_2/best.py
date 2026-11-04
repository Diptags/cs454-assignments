"""
best.py -- the single best SBFL formula evolved by our GP (gp.py)
=================================================================

How it was obtained
-------------------
Final GP run trained on all 119 bugs (gp.py --train-all, seed 401,
population 300, 60 generations, fitness = maximise acc@1, then minimise wef).
The formula is hard-coded here; GP is NOT re-run by this file.

Results on the 119 provided bugs (method level, max aggregation)
----------------------------------------------------------------
    acc@1 = 53/119 with pessimistic ties (76/119 with optimistic ties)
    wef   = 4.83   with pessimistic ties (4.17 with average ties)
    Ochiai, for comparison: 46/119 (70 optimistic), wef 5.09

Requirements of the assignment respected
----------------------------------------
* signature is exactly  sbfl(ep, ef, np, nf)
* called with plain Python ints, one spectrum at a time (no numpy)
* division uses the assignment's safe div (x / 0 = 1)
"""
import math


def div(x: float, y: float) -> float: # divide x with y
    """Safe division from the assignment skeleton: x / 0 is defined as 1."""
    if y == 0:
        return 1.0
    else:
        return x / y


def psqrt(x: float) -> float: # protected square root used by the GP
    """sqrt(|x|): the GP's square root, protected against negative inputs."""
    return math.sqrt(abs(x))


def sbfl(ep: int, ef: int, np: int, nf: int) -> float:
    """Suspiciousness of one statement.

    Args:
        ep: number of passing tests that execute the statement
        ef: number of failing tests that execute the statement
        np: number of passing tests that do NOT execute the statement
        nf: number of failing tests that do NOT execute the statement
    Returns:
        a float; higher = more suspicious.  Method score = max over its
        statements (done by the caller).

    Behaviour (see the report, Section 5):
      * one failing test (most bugs): the score decreases as ep grows, i.e.
        statements executed by fewer passing tests are more suspicious;
      * several failing tests: statements executed by ALL failing tests
        (nf = 0) dominate, and a large ep is penalised strongly.  Terms such
        as (ef - ep) also create a few data-specific peaks (e.g. with 3
        failing tests, ep = 1-2 can score above ep = 0); this is a pattern
        learned from these projects, not a general rule.

    Readable form ("/" = safe div, sqrt = psqrt):
    ((sqrt((sqrt(((nf - ep) + sqrt(np))) * (ef * ef))) / (sqrt((sqrt((nf - ef)) + ((nf - ep) * (ef - ep)))) * ((1 * (ef + nf)) / nf))) / sqrt((sqrt((nf - ef)) + ((((1 * 1) + nf) - ep) * (ef - (1 + nf))))))
    """
    suspiciousness = div(div(psqrt((psqrt(((nf - ep) + psqrt(np))) * (ef * ef))), (psqrt((psqrt((nf - ef)) + ((nf - ep) * (ef - ep)))) * div((1.0 * (ef + nf)), nf))), psqrt((psqrt((nf - ef)) + ((((1.0 * 1.0) + nf) - ep) * (ef - (1.0 + nf))))))
    return suspiciousness
