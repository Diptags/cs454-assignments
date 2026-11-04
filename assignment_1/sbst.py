"""Search based test input generation for a subset of Python.

usage: python sbst.py examples/example1.py [--algorithm hc|avm|sa|tabu|random] [--seed N]

Vocabulary used throughout this file:
  - predicate: the condition of an if / while, e.g. `x == 42`.
  - branch id (bid): an integer that identifies one if / while / for statement.
  - target: a pair (bid, outcome). Every branching statement has two targets,
    (bid, True) and (bid, False), i.e. "the predicate evaluated to True" and
    "the predicate evaluated to False". Covering all targets = branch coverage.
  - branch distance: how far a predicate is from taking a given outcome.
    0 means it already takes that outcome; bigger means further away.
  - approach level: how many control dependencies (enclosing conditions) were
    still missed when the execution diverged away from the target.
  - fitness: approach_level + normalise(branch_distance). Lower is better,
    0 means the target is covered.

Small stateless helper functions live in sbst_helpers.py.

Table of contents (the file reads top to bottom in the order the program runs):

  0. CONFIGURATION ........ constants
  1. MAIN ................. main(): the whole pipeline in one place
  2. INSTRUMENTATION ...... Branch, Instrumenter      (rewrite the AST, once)
  3. RUNTIME RECORDER ..... Runtime                   (called by the rewritten code)
  4. EXECUTION ............ Abort, Result, Executor   (run one input)
  5. FITNESS .............. normalise(), fitness()    (score one run for one target)
  6. SEARCH ............... Search                    (hc, avm, sa, tabu, random)
  7. TEST FILE OUTPUT ..... render_tests()            (write test_<file>.py)
"""
import argparse
import ast
import collections
import contextlib
import io
import math
import os
import random
import signal

from sbst_helpers import (
    K, RT,                                              # shared constants
    rt_call, thunk, terminates,                         # used in section 2
    empty_distance,                                     # used in section 3
)


# ═══════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
#
# K and RT are shared with sbst_helpers.py and are defined there.
# ═══════════════════════════════════════════════════════════════════════════

STEP_LIMIT = 100_000     # max predicate evaluations / loop iterations in a single run
TIME_LIMIT = 0.5         # seconds allowed for a single run
RANGES = (10, 100, 1000)  # random inputs are drawn from [-r, r] for a randomly chosen r
STEPS = (1, 10, 100, 1000, 10_000, 100_000)  # neighbourhood step sizes (hc, sa, tabu)
EPS = 1e-6               # smallest non-zero fitness, so only real coverage gets fitness 0

SA_START_TEMP = 1.0      # simulated annealing: initial temperature
SA_COOLING = 0.995       # simulated annealing: temperature is multiplied by this each step
SA_MIN_TEMP = 0.01       # simulated annealing: below this the search is "frozen" -> restart
SA_APPROACH_WEIGHT = 100  # simulated annealing: energy cost of one approach level
TABU_SIZE = 50           # tabu search: how many recently visited inputs are forbidden
TABU_PATIENCE = 5        # tabu search: restart after this many moves without a new best

# --algorithm value -> the Search method that implements it
ALGORITHMS = {
    "hc": "search_hill_climbing",
    "avm": "search_avm",
    "sa": "search_simulated_annealing",
    "tabu": "search_tabu",
    "random": "search_random",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Command-line entry point: instrument the file, search, and write the tests.

    Also prints, per function, how many branch targets were covered and how many
    fitness evaluations it took (numbers useful for the report).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="the target python file to generate unit tests for")
    parser.add_argument("--algorithm", choices=list(ALGORITHMS), default="hc",
                        help="hc = hill climbing (default), avm = alternating variable "
                             "method, sa = simulated annealing, tabu = tabu search, "
                             "random = random search")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--budget", type=int, default=20_000,
                        help="max fitness evaluations spent on each branch target")
    parser.add_argument("--init", type=int, default=10,
                        help="random inputs evaluated per function before targeted search")
    args = parser.parse_args()

    with open(args.target, "r") as f:
        code = f.read()

    # Section 2: parse the file and rewrite its AST in memory
    path = os.path.abspath(args.target)
    module = os.path.basename(args.target).removesuffix(".py")
    tree = ast.parse(code, filename=path)

    # Give log for this step, because it may take some time to run the program
    print(f"Target: {args.target} ({len(tree.body)} top-level statements.")
    print(f"Instrumenting {args.target} ({len(tree.body)} top-level statements, "
          f"{sum(isinstance(s, ast.FunctionDef) for s in tree.body)} functions)")
    print("=" * 80)

    # Setting up the instrumenter and instrumenting the AST
    inst = Instrumenter()
    tree = inst.instrument(tree)

    # Section 4: load the rewritten module so its functions can be called
    executor = Executor(tree, path, module)
    rng = random.Random(args.seed)

    # Give log for this step, because it may take some time to run the program
    print(f"Instrumented {len(inst.branches)} branches in "
          f"{len(inst.functions)} functions, using random seed {args.seed}")
    print(f"Searching for inputs to cover {len(inst.branches)} branches in "
          f"{len(inst.functions)} functions, using {args.algorithm} search")
    print("=" * 80)

    # Section 6: search for inputs, one function at a time
    suites = []
    total_targets = total_covered = 0
    for name, fdef in inst.functions.items():
        n_args = len(fdef.args.args)
        search = Search(executor, inst.branches, name, n_args, rng)
        before = executor.evaluations
        search.run(args.algorithm, args.budget, args.init)
        tests = search.minimal_tests()
        suites.append((name, tests))

        hit = [t for t in search.targets if t in search.covered]
        total_targets += len(search.targets)
        total_covered += len(hit)
        print(f"{name}: {len(hit)}/{len(search.targets)} branches covered, "
              f"{len(tests)} tests, {executor.evaluations - before} fitness evaluations")
        for bid, outcome in search.targets:
            if (bid, outcome) not in search.covered:
                b = inst.branches[bid]
                print(f"  not covered: line {b.lineno} ({b.kind}) -> {outcome}")

    print(f"total: {total_covered}/{total_targets} branches, "
          f"{executor.evaluations} fitness evaluations")
    print("=" * 80)

    # Section 7: write test_<target>.py next to the target file
    test_file_name = os.path.join(os.path.dirname(args.target), "test_" + os.path.basename(args.target))
    with open(test_file_name, "w") as f:
        f.write(render_tests(module, suites))
    print("Test suite generated in", test_file_name)


# ═══════════════════════════════════════════════════════════════════════════
# 2. INSTRUMENTATION
#
# Runs once, before any search. We walk the AST of the target file and for
# each if / while / for we
#   1. give it a new branch id,
#   2. remember its control dependencies (for the approach level), and
#   3. replace its test / iterable with calls into the Runtime (section 3).
#
# After instrumentation, a predicate such as `x == 42` becomes
#     __sbst_rt__.record_branch(0, __sbst_rt__.distance_compare("==", x, 42))
# ═══════════════════════════════════════════════════════════════════════════

# maps AST comparison operator classes to the strings Runtime.distance_compare understands
OPS = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.In: "in",
}


class Branch:
    """Static information about one branching statement.

    `deps` is the list of conditions that must hold for execution to reach this
    statement, from the outermost to the innermost. For example in

        def foo(x, y):
            if x == 42:          # branch 0, deps = []
                if y == 0:       # branch 1, deps = [(0, True)]

    reaching branch 1 requires branch 0 to be True.
    """

    def __init__(self, bid, kind, lineno, func, deps):
        """Store the branch id, kind, source line, owning function and dependencies."""
        self.bid = bid
        self.kind = kind      # "if", "while" or "for"
        self.lineno = lineno
        self.func = func      # name of the top level function containing it
        self.deps = deps      # [(branch id, required outcome), ...] from outermost to innermost


class Instrumenter:
    """Walks the module, registers branches with their control dependencies, and
    rewrites predicates in place so that they report branch distances."""

    def __init__(self):
        """Start with no branches and no functions discovered."""
        self.branches = {}    # branch id -> Branch
        self.functions = {}   # name -> ast.FunctionDef (top level only)
        self._func = None     # name of the function currently being walked

    # ── entry point ──────────────────────────────────────────────────────────

    def instrument(self, tree):
        """Instrument every top-level function of `tree` and return the modified tree.

        Only top-level `def`s are test targets (the assignment says all inputs
        are the function's int arguments). `fix_missing_locations` fills in line
        numbers for the new nodes we created, which `compile()` requires.
        """
        for stmt in tree.body:
            if isinstance(stmt, ast.FunctionDef):
                self.functions[stmt.name] = stmt
                self._func = stmt.name
                self._block(stmt.body, [])
        self._func = None
        return ast.fix_missing_locations(tree)

    # ── walking statements ───────────────────────────────────────────────────

    def _block(self, stmts, deps):
        """Walk a list of statements, instrumenting each one in order.

        `deps` grows as we walk: an `if` whose body always exits (return/break/...)
        adds a control dependency for the statements that follow it.

        `match` nodes are spliced out and replaced with an if-elif-else chain in-place
        before `_stmt` is called. We use a `while` loop (not `for`) because the splice
        `stmts[i:i+1] = replacement` modifies the live list, and `continue` lets us
        re-visit the same index to process the freshly inserted `If` node.
        """
        deps = list(deps)
        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            if isinstance(stmt, ast.Match):
                replacement = self._match_to_if(stmt)
                if replacement is not None:
                    stmts[i:i + 1] = replacement
                    continue  # re-visit index i, now holding the replacement If
                # unsupported patterns: keep the match itself, instrument the case bodies
                for case in stmt.cases:
                    self._block(case.body, deps)
                i += 1
                continue
            self._stmt(stmt, deps)
            if isinstance(stmt, ast.If):
                bid = stmt._sbst_bid
                body_exits, else_exits = terminates(stmt.body), terminates(stmt.orelse)
                if body_exits and not else_exits:
                    deps.append((bid, False))
                elif else_exits and not body_exits:
                    deps.append((bid, True))
            i += 1

    def _stmt(self, node, deps):
        """Instrument one statement and recurse into the blocks it contains.

        - if / while: the test becomes `record_branch(bid, <predicate with distances>)`;
          the body needs (bid, True). For `if`, the else part needs (bid, False).
          Note that `elif` is just an `if` inside the else part, so it naturally
          gets a (parent, False) dependency from the enclosing if's orelse.
        - for: the iterable is wrapped with `record_loop(bid, <iterable>)` which yields
          exactly the same items but records whether the body was entered.
        - match: NOT handled here. _block replaces match with if-elif-else before
          calling _stmt, so _stmt only ever sees ast.If, not ast.Match.
        Any other statement (assignment, return, expression, ...) needs no
        instrumentation and is silently ignored.
        """
        if isinstance(node, (ast.If, ast.While)):
            bid = self._new_branch(node, "if" if isinstance(node, ast.If) else "while", deps)
            node._sbst_bid = bid
            node.test = rt_call("record_branch", ast.Constant(bid), self._pred(node.test))
            self._block(node.body, deps + [(bid, True)])
            if isinstance(node, ast.If):  # the else / elif part
                self._block(node.orelse, deps + [(bid, False)])
        elif isinstance(node, ast.For):
            bid = self._new_branch(node, "for", deps)
            node.iter = rt_call("record_loop", ast.Constant(bid), node.iter)
            self._block(node.body, deps + [(bid, True)])

    def _new_branch(self, node, kind, deps):
        """Register a new branching statement and return its id (0, 1, 2, ...)."""
        bid = len(self.branches)
        self.branches[bid] = Branch(bid, kind, node.lineno, self._func, list(deps))
        return bid

    # ── rewriting predicates ─────────────────────────────────────────────────

    def _pred(self, e):
        """Turn a boolean expression into one evaluating to (value, d_true, d_false).

        Recursively rewrites the predicate, e.g.

            x > 0 and not y == 3
        becomes
            __sbst_rt__.distance_and(
                lambda: __sbst_rt__.distance_compare(">", x, 0),
                lambda: __sbst_rt__.distance_not(
                    __sbst_rt__.distance_compare("==", y, 3)))

        Anything that is not and / or / not / a single comparison (a variable, a
        function call, arithmetic, ...) is handed to `distance_truthy`.
        """
        if isinstance(e, ast.BoolOp):
            attr = "distance_and" if isinstance(e.op, ast.And) else "distance_or"
            return rt_call(attr, *[thunk(self._pred(v)) for v in e.values])
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
            return rt_call("distance_not", self._pred(e.operand))
        if isinstance(e, ast.Compare) and len(e.ops) == 1 and type(e.ops[0]) in OPS:
            op = OPS[type(e.ops[0])]
            return rt_call("distance_compare", ast.Constant(op), e.left, e.comparators[0])
        if isinstance(e, ast.IfExp):
            # Ternary `A if C else B` used as a predicate.
            # We recurse into all three sub-expressions so distance_ternary() receives a
            # (value, d_true, d_false) triple for each part and can compute the
            # gradient through C (see Runtime.distance_ternary for the distance formula).
            # Each part is wrapped in a lambda so it is evaluated at runtime with
            # the live argument values, not here at instrumentation time.
            #
            # Example: `True if x == 42 else False` with x = 10
            #   C thunk → distance_compare("==", 10, 42) → (False, 32, 0)
            #   distance_ternary() sees C is False → d_true = c_dt + a_dt = 32 + 0 = 32
            #   Without this, distance_truthy() would give a flat d_true = K for any x.
            return rt_call("distance_ternary",
                         thunk(self._pred(e.test)),
                         thunk(self._pred(e.body)),
                         thunk(self._pred(e.orelse)))
        return rt_call("distance_truthy", e)

    # ── match-case support ───────────────────────────────────────────────────

    def _match_to_if(self, node):
        """Convert a match statement into an equivalent if-elif-else chain.

        `match` has no single `node.test` to replace, so we rewrite the whole
        statement before the instrumentation pass sees it:

          match y:               →   if y == 5:          (bid B0, deps=[])
              case 5:    body1           body1
              case 11:   body2       elif y == 11:        (bid B1, deps=[(B0,False)])
              case -57|43: body3         body2
              case _:    body4       elif y==-57 or y==43:(bid B2, deps=[(B0,F),(B1,F)])
                                         body3
                                     else:
                                         body4

        We iterate in reverse and build the nested If from the inside out, because in
        the AST `elif` is just an `if` inside the `orelse` of the parent `if`.
        After substitution the normal _stmt / _block machinery instruments each `if`,
        so each case automatically gets the correct branch id and approach-level deps.

        Returns None if any case cannot be rewritten without changing its meaning
        (a guard `case 5 if x > 0`, a capture `case y`, a sequence/class pattern, ...).
        The caller then keeps the original match statement.
        """
        wildcard_body = []
        normal_cases = []
        for case in node.cases:
            if case.guard is not None:
                return None
            if isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None:
                if case.pattern.name is not None:
                    return None  # `case y:` binds a name, which an `else` cannot do
                wildcard_body = list(case.body)  # `case _:`
            else:
                normal_cases.append(case)

        tests = [self._pattern_to_cmp(node.subject, case.pattern) for case in normal_cases]
        if any(t is None for t in tests):
            return None
        if not normal_cases:
            return wildcard_body or [ast.Pass()]

        # Build nested If from last case to first; each iteration wraps the previous tail
        current_orelse = wildcard_body
        for case, test_expr in reversed(list(zip(normal_cases, tests))):
            if_node = ast.If(test=test_expr, body=list(case.body),
                             orelse=list(current_orelse))
            ast.copy_location(if_node, node)
            ast.copy_location(test_expr, node)
            current_orelse = [if_node]

        return current_orelse  # [outermost If]

    def _pattern_to_cmp(self, subject, pattern):
        """Convert a match-case pattern into an equivalent comparison AST node.

          Pattern in source    AST node      Result
          ─────────────────────────────────────────────────────────────────
          case 5:              MatchValue    subject == 5
          case -57 | 43:       MatchOr       subject == -57 or subject == 43
          case _:              MatchAs(None) (wildcard — never passed here)

        For `case -57`, the value is stored as UnaryOp(USub, Constant(57)) in the
        AST; we pass it directly as a comparator so `subject == -57` compiles
        correctly without any special handling.

        Returns None for any other pattern, so the caller keeps the original match.
        """
        if isinstance(pattern, ast.MatchValue):
            return ast.Compare(left=subject, ops=[ast.Eq()], comparators=[pattern.value])
        if isinstance(pattern, ast.MatchOr):
            parts = [self._pattern_to_cmp(subject, p) for p in pattern.patterns]
            if any(p is None for p in parts):
                return None
            return ast.BoolOp(op=ast.Or(), values=parts)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 3. RUNTIME RECORDER
#
# The rewritten code from section 2 calls these methods while it runs.
# The distance_* methods compute branch distances; record_branch / record_loop
# store them and return the ordinary value, so the program behaves exactly as before.
# ═══════════════════════════════════════════════════════════════════════════

class Runtime:
    """Records, for one execution, the minimum branch distances seen at every branch.

    One Runtime object lives inside the instrumented module under the global name
    `__sbst_rt__`. Before every execution of the target function it is `reset()`,
    and after the execution its `dist` and `taken` fields describe what happened.

    All distance_* methods (distance_compare, distance_in, distance_and, distance_or,
    distance_not, distance_ternary, distance_truthy) return a triple
        (value, d_true, d_false)
    where `value` is the real result of the predicate, `d_true` is the distance to
    making it True and `d_false` is the distance to making it False. Exactly one
    of the two distances is 0 (the outcome that actually happened).
    """

    # ── bookkeeping ──────────────────────────────────────────────────────────

    def __init__(self):
        """Create an empty recorder."""
        self.reset()

    def reset(self):
        """Forget everything recorded so far; called before each execution."""
        self.dist = {}      # branch id -> [min distance to True, min distance to False]
        self.taken = set()  # (branch id, outcome) pairs actually executed
        self.steps = 0

    def tick(self):
        """Count one predicate evaluation / loop iteration, aborting endless loops.

        Some inputs make the target loop forever (e.g. example2 with a = 300,
        b = 400, c = 0: neither branch in the loop changes `a`). Without this guard
        the search itself would hang.
        """
        self.steps += 1
        if self.steps > STEP_LIMIT:
            raise Abort()

    def observe(self, bid, d_true, d_false):
        """Store the distances of branch `bid`, keeping the minimum over the run.

        A `while` predicate, or an `if` inside a loop, is evaluated many times in one
        execution. For fitness we care about the closest the execution ever got, so
        we keep the smallest distance seen for each outcome.
        """
        rec = self.dist.get(bid)
        if rec is None:
            self.dist[bid] = [d_true, d_false]
        else:
            rec[0] = min(rec[0], d_true)
            rec[1] = min(rec[1], d_false)

    # ── branch points: called at every if / while / for ──────────────────────

    def record_branch(self, bid, d):
        """Called at every `if` / `while` test. Records the result and returns the bool.

        `d` is the (value, d_true, d_false) triple produced by the predicate helpers.
        We remember the distances (for fitness) and which outcome was taken (for
        coverage), then return the plain boolean so the program runs as normal.
        """
        v, dt, df = d
        self.tick()
        self.observe(bid, dt, df)
        self.taken.add((bid, bool(v)))
        return bool(v)

    def record_loop(self, bid, iterable):
        """Wrap the iterable of a `for` loop so we can see whether the body runs.

        `for i in range(n):` becomes `for i in __sbst_rt__.record_loop(bid, range(n)):`.
        We call iter() right away so errors (e.g. iterating over an int) happen at
        the same moment as in the original program.
        """
        it = iter(iterable)
        return self._record_loop_items(bid, iterable, it)

    def _record_loop_items(self, bid, iterable, it):
        """Generator doing the actual recording for `loop`.

        For a `for` loop the two targets are:
          - True:  the body is executed at least once
          - False: the loop finishes by running out of items (not via break/return)
        If the body never runs and the iterable is a range, `empty_distance` tells
        the search how far the range is from being non-empty (e.g. range(-5) -> 6).
        """
        entered = False
        for item in it:
            self.tick()
            if not entered:
                entered = True
                self.observe(bid, 0, K)
                self.taken.add((bid, True))
            yield item
        # reaching here means the iterator was exhausted, i.e. the loop exit branch
        if not entered:
            self.observe(bid, empty_distance(iterable), 0)
        else:
            self.observe(bid, 0, 0)
        self.taken.add((bid, False))

    # ── predicate distances: each returns (value, d_true, d_false) ───────────

    def distance_compare(self, op, a, b):
        """Evaluate a single comparison `a op b` and compute its branch distances.

        The formulas (Korel / Tracey style), with K = 1:
            a == b   true: 0 / false: |a - b|      (to flip a true one: K)
            a != b   true: 0 / false: K            (to flip a true one: |a - b|)
            a <  b   false -> d_true = a - b + K   (need to go strictly below b)
            a <= b   false -> d_true = a - b
            a >  b   false -> d_true = b - a + K
            a >= b   false -> d_true = b - a
            a in c   false -> d_true = min |a - e| over the elements e of c
                     true  -> d_false = K
        For ordering operators `d` below is |a - b|, which equals the signed values
        above whenever the comparison is on the wrong side.

        Examples:
            distance_compare("==", 10, 42)         -> (False, 32, 0)
            distance_compare("<", 5, 3)            -> (False, 3, 0)   # 5 - 3 + 1
            distance_compare("in", 40, [42, 3817]) -> (False, 2, 0)   # closest is 42
        """
        if op == "in":
            return self.distance_in(a, b)
        d = abs(a - b)
        if op == "==":
            v = a == b
            return (v, 0, K) if v else (v, d, 0)
        if op == "!=":
            v = a != b
            return (v, 0, d) if v else (v, K, 0)
        if op in ("<", "<=", ">", ">="):
            if op == "<":
                v = a < b
            elif op == "<=":
                v = a <= b
            elif op == ">":
                v = a > b
            else:
                v = a >= b
            # distance from the boundary; strict side needs an extra K to cross it
            strict = op in ("<", ">")
            if v:
                # currently True: distance to False is how far we are from the boundary
                return (v, 0, d if strict else d + K)
            return (v, d + K if strict else d, 0)

    def distance_in(self, a, collection):
        """`a in collection`: the distance to True is the distance to the closest element.

        Example: 40 in [42, 3817, 1038472] is False; the closest element is 42, so
        d_true = |40 - 42| = 2 and the search learns to move `a` towards 42.
        An empty collection gives d_true = K.
        """
        v = a in collection
        if v:
            return (v, 0, K)
        return (v, min((abs(a - e) for e in collection), default=K), 0)

    def distance_and(self, *thunks):
        """`A and B and ...` with short-circuit evaluation.

        Each thunk is a lambda that returns the (value, d_true, d_false) triple of
        one operand. Lambdas are needed so an operand is only evaluated if Python
        would evaluate it (e.g. `x != 0 and 10 / x > 1` must not divide by zero).

        Distances:
          - to make the `and` True, every operand must be True, so the distances
            add up. We only know the distance of the first False operand; the ones
            after it were skipped, so each gets a penalty of K.
          - to make it False, it is enough that one operand becomes False, so
            d_false is the smallest d_false among the operands.
        """
        min_false = math.inf
        for i, t in enumerate(thunks):
            v, dt, df = t()
            if not v:
                # operands after a False one are never evaluated: penalise each with K
                return (False, dt + K * (len(thunks) - 1 - i), 0)
            min_false = min(min_false, df)
        return (True, 0, min_false)

    def distance_or(self, *thunks):
        """`A or B or ...` with short-circuit evaluation; mirror image of `distance_and`.

        - to make it True, one operand is enough: d_true is the minimum d_true.
        - to make it False, all operands must be False: the distances add up, and
          skipped operands (after the first True one) are penalised with K each.
        """
        min_true = math.inf
        for i, t in enumerate(thunks):
            v, dt, df = t()
            if v:
                return (True, 0, df + K * (len(thunks) - 1 - i))
            min_true = min(min_true, dt)
        return (False, min_true, 0)

    def distance_not(self, d):
        """`not A`: flip the value and swap the two distances.

        Being close to making A True is the same as being close to making
        `not A` False, and vice versa.
        """
        v, dt, df = d
        return (not v, df, dt)

    def distance_ternary(self, cond_thunk, body_thunk, orelse_thunk):
        """`A if C else B` used as a predicate — the ternary (conditional) expression.

        Python only evaluates one of A or B depending on C, but to keep the gradient
        we always evaluate both eagerly. The distance formula follows the taken path
        but still "looks through" the condition to guide the search:

          If C is True (using A):
            d_true  = a_dt                   (C already True, need A truthy)
            d_false = min(c_df + b_df, a_df) (flip C then need B falsy, OR flip A)

          If C is False (using B):
            d_true  = c_dt + a_dt            (flip C then need A truthy — the natural path)
            d_false = b_df                   (C already False, need B falsy)

        Example: `True if x == 42 else False` with x = 10
          C: distance_compare("==", 10, 42) -> (False, 32, 0); c_dt=32
          A: distance_truthy(True)      -> (True,  0,  K)
          B: distance_truthy(False)     -> (False, K,  0)
          C is False -> d_true = 32 + 0 = 32  ← gradient towards x=42 is preserved
        """
        c_v, c_dt, c_df = cond_thunk()
        # Evaluate both sides eagerly so we have the "flip C" gradient in both directions
        a_v, a_dt, a_df = body_thunk()
        b_v,    _, b_df = orelse_thunk()
        if c_v:
            return (a_v, a_dt, min(c_df + b_df, a_df))
        else:
            return (b_v, c_dt + a_dt, b_df)

    def distance_truthy(self, value):
        """Distances for a plain value used as a condition, e.g. `if x:` or `if f(y):`.

        A number (or bool) is False only when it is 0, so:
          - distance to True  (when x == 0): K
          - distance to False (when x != 0): |x|      (1 for True)
        """
        v = bool(value)
        return (v, 0, abs(value)) if v else (v, K, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 4. EXECUTION
#
# Load the instrumented module in memory and run one input through it.
# ═══════════════════════════════════════════════════════════════════════════

class Abort(BaseException):
    """Raised inside the target to stop runs that loop too long.

    Derives from BaseException so that `except Exception` in the target cannot swallow it.
    """


class Result:
    """Everything we learned from executing the target function once."""

    def __init__(self, args, status, value, dist, taken):
        """Bundle the input, how the run ended, and the recorded branch data."""
        self.args = args
        self.status = status  # "ok", "exc" or "abort"
        self.value = value    # return value, or exception class for "exc"
        self.dist = dist      # branch id -> [min d_true, min d_false] (see Runtime)
        self.taken = taken    # set of (branch id, outcome) covered by this run


class Executor:
    """Loads the instrumented module in memory and runs its functions.

    The instrumented code is never written to disk: we `compile` the modified
    AST and `exec` it into a fresh dictionary that acts as the module's globals.
    The Runtime is placed in that dictionary under the name `__sbst_rt__`, which
    is how the instrumented code finds it.
    """

    def __init__(self, tree, path, module_name):
        """Compile and execute the instrumented module; its output is hidden."""
        self.rt = Runtime()
        self.namespace = {"__name__": module_name, "__file__": path, RT: self.rt}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(tree, path, "exec"), self.namespace)
        self.evaluations = 0  # number of fitness evaluations (= executions) so far

    def run(self, func, args):
        """Execute `func(*args)` once and return a Result.

        - the Runtime is reset first, so the result only describes this run
        - a timer aborts runs slower than TIME_LIMIT (e.g. huge arithmetic)
        - output printed by the target is swallowed so the console stays clean
        - status "abort": the run was stopped (endless loop, timeout, deep
          recursion); such inputs are never written as tests, because the test
          would hang or crash pytest
        - status "exc": the target raised an ordinary exception; this is still a
          valid test, written with `pytest.raises`
        """
        self.evaluations += 1
        self.rt.reset()
        status, value = "ok", None
        old = signal.signal(signal.SIGALRM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, TIME_LIMIT)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                value = self.namespace[func](*args)
        except (Abort, RecursionError):
            status = "abort"
        except Exception as ex:
            status, value = "exc", type(ex)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
        return Result(tuple(args), status, value, self.rt.dist, set(self.rt.taken))


# ── signal handler for Executor.run (kept here because it raises Abort) ──

def _on_timeout(_signum, _frame):
    """Signal handler for the per-run timer: abort the current run.

    Python's signal API always passes (signal number, stack frame); we need neither.
    """
    raise Abort()


# ═══════════════════════════════════════════════════════════════════════════
# 5. FITNESS
#
# Turn one Result into a single number for one target: lower is better.
# ═══════════════════════════════════════════════════════════════════════════

def fitness(result, target, branches):
    """approach level + normalised branch distance for `target` = (branch id, outcome).

    Walks the path [outer dependencies..., target] from the inside out and stops
    at the deepest node the execution actually reached:
      - approach level = number of path nodes after that node (not reached)
      - branch distance = that node's distance to the outcome the path requires

    Worked example, foo in example1, target "y == 0 is True" (path:
    x == 42 True -> y == 0 True):
      foo(10, 5): only `x == 42` reached -> approach 1, distance |10-42| = 32
                  fitness = 1 + 32/33 = 1.97
      foo(42, 5): `y == 0` reached -> approach 0, distance |5-0| = 5
                  fitness = 0 + 5/6 = 0.83
      foo(42, 0): target taken -> fitness 0

    An aborted run never gets 0 (EPS at least), because we cannot use it as a test.
    If nothing on the path was reached (e.g. an exception earlier), the fitness
    is the worst possible value for this path, len(path).
    """
    if is_covered(result, target):
        return 0.0
    approach, d = fitness_parts(result, target, branches)
    if d is None:
        return approach
    return max(approach + normalise(d), EPS)


def fitness_parts(result, target, branches):
    """The two ingredients of the fitness: (approach level, raw branch distance).

    Walks the path [outer dependencies..., target] from the inside out and stops
    at the deepest node the execution actually reached. Returns (len(path), None)
    if nothing on the path was reached.
    Used by fitness() above, and by simulated annealing, which needs the raw
    distance instead of the normalised one (see Search.sa_energy).
    """
    path = branches[target[0]].deps + [target]
    for j in range(len(path) - 1, -1, -1):
        b, o = path[j]
        rec = result.dist.get(b)
        if rec is not None:
            return len(path) - 1 - j, (rec[0] if o else rec[1])
    return len(path), None


def is_covered(result, target):
    """True if this run took `target` and can be used as a test (was not aborted)."""
    return target in result.taken and result.status != "abort"


def normalise(d):
    """Map a branch distance in [0, inf) to [0, 1): 0 -> 0, 1 -> 0.5, 32 -> 0.97.

    Keeping it below 1 guarantees that one approach level always outweighs any
    branch distance, so getting one condition deeper is always an improvement.
    """
    return d / (d + 1)


# ═══════════════════════════════════════════════════════════════════════════
# 6. SEARCH
#
# For one function: find inputs that cover its branch targets.
# ═══════════════════════════════════════════════════════════════════════════

class Search:
    """Search for inputs covering every branch target of one function.

    Keeps track of which targets are already covered (by any input, not only the
    one aimed at them: "collateral coverage"), and of the inputs worth keeping as
    test cases.
    """

    def __init__(self, executor, branches, func, n_args, rng):
        """Set up the search for function `func`, which takes `n_args` int arguments."""
        self.ex = executor
        self.branches = branches
        self.func = func
        self.n_args = n_args
        self.rng = rng
        # two targets per branching statement of this function
        self.targets = [(b.bid, o) for b in branches.values() if b.func == func
                        for o in (True, False)]
        self.covered = set()
        self.tests = []   # results that are usable as test cases
        self.best = {}    # target -> (fitness, args): best input seen so far

    # ── overall strategy ─────────────────────────────────────────────────────

    def run(self, algorithm, budget, n_init):
        """Cover as many targets of the function as possible.

        First a few random inputs pick off the easy targets. Then each remaining
        target is attacked one by one with the chosen algorithm (a key of
        ALGORITHMS, e.g. "hc"). Targets covered along the way (collateral
        coverage) are skipped.
        """
        if self.n_args == 0:
            self.evaluate([])
            return
        search_algorithm = getattr(self, ALGORITHMS[algorithm])
        for _ in range(n_init):
            self.evaluate(self.random_input())
        # shallow targets first: they often cover deeper ones collaterally
        for t in sorted(self.targets, key=lambda t: len(self.branches[t[0]].deps)):
            if t in self.covered:
                continue
            search_algorithm(t, budget)

    # ── algorithms ───────────────────────────────────────────────────────────
    # Every search_* method has the same contract:
    #   - aim at one `target` and stop as soon as it is covered
    #   - spend at most (about) `budget` fitness evaluations
    #   - return True if the target ends up covered

    def search_hill_climbing(self, target, budget):
        """Steepest-descent hill climbing with random restarts.  (--algorithm hc)

        1. start from the best input seen so far for this target (or a random one)
        2. evaluate all neighbours, move to the one with the lowest fitness
        3. repeat until fitness is 0 (covered) or no neighbour is better
           (a local optimum), in which case restart from a random input
        4. give up after `budget` fitness evaluations: the target is then
           reported as not covered (possibly unreachable)
        """
        start = self.ex.evaluations
        x = self.start_point(target)
        while self.ex.evaluations - start < budget and target not in self.covered:
            f = self.fit(x, target)
            # steepest descent: move to the best neighbour until none improves
            while f > 0 and self.ex.evaluations - start < budget:
                best_n, best_f = None, f
                for n in self.neighbours(x):
                    nf = self.fit(n, target)
                    if nf < best_f:
                        best_n, best_f = n, nf
                        if nf == 0:
                            break
                if best_n is None:
                    break  # local optimum
                x, f = best_n, best_f
            if f == 0:
                return True
            x = self.random_input()  # restart
        return target in self.covered

    def search_avm(self, target, budget):
        """Alternating Variable Method (Korel, 1990).  (--algorithm avm)

        Optimise ONE variable at a time, keeping the others fixed:
          1. exploratory move: try x[i] - 1 and x[i] + 1
          2. if one of them improves the fitness, keep going in that direction
             with pattern moves whose step DOUBLES each time (2, 4, 8, ...) until
             the fitness stops improving; then go back to step 1 for the same x[i]
          3. if neither exploratory move improves, move on to the next variable
          4. if no variable can be improved, we are in a local optimum: restart
             from a random input

        Doubling steps make AVM fast for integers: from x = 10 to x = 42 it
        needs +1, +2, +4, +8, +16 (-> 41), then +1 (-> 42): about 8 evaluations,
        instead of hill climbing's full neighbourhood (12 inputs per variable)
        at every step.
        """
        start = self.ex.evaluations

        def out_of_budget():
            return self.ex.evaluations - start >= budget

        x = self.start_point(target)
        f = self.fit(x, target)
        i, failed_in_a_row = 0, 0
        while f > 0 and not out_of_budget() and target not in self.covered:
            x, f, improved = self._avm_optimise_variable(x, f, i, target, out_of_budget)
            if improved:
                failed_in_a_row = 0          # stay on the same variable
            else:
                failed_in_a_row += 1
                i = (i + 1) % self.n_args    # try the next variable
            if failed_in_a_row >= self.n_args:
                x = self.random_input()      # no variable helps: restart
                f = self.fit(x, target)
                failed_in_a_row = 0
        return target in self.covered

    def _avm_optimise_variable(self, x, f, i, target, out_of_budget):
        """One round of AVM on variable i: exploratory moves, then pattern moves.

        Returns (new x, new fitness, whether anything improved).
        """
        for direction in (-1, 1):
            y = list(x)
            y[i] += direction
            fy = self.fit(y, target)
            if fy < f:
                x, f, step = y, fy, direction
                # pattern moves: keep going in the same direction, doubling the step
                while f > 0 and not out_of_budget():
                    step *= 2
                    z = list(x)
                    z[i] += step
                    fz = self.fit(z, target)
                    if fz >= f:
                        break
                    x, f = z, fz
                return x, f, True
        return x, f, False

    def search_simulated_annealing(self, target, budget):
        """Simulated annealing with restarts.  (--algorithm sa)

        Like hill climbing, but it looks at ONE random neighbour per step and may
        also accept a WORSE neighbour, with probability exp(-(worse by) / T).
        The temperature T starts at SA_START_TEMP and is multiplied by SA_COOLING
        after every step, so bad moves are accepted often at the start (to escape
        local optima) and almost never at the end (to settle into a minimum).
        When T drops below SA_MIN_TEMP the search is "frozen" and restarts from a
        random input with the temperature reset.

        "Worse by" is measured with sa_energy (log scale), not with the normal
        fitness; see sa_energy for why.
        """
        start = self.ex.evaluations
        x = self.start_point(target)
        e = self.sa_energy(x, target)
        temperature = SA_START_TEMP
        while self.ex.evaluations - start < budget and target not in self.covered:
            y = self.random_neighbour(x)
            ey = self.sa_energy(y, target)
            worse_by = ey - e
            if worse_by <= 0 or self.rng.random() < math.exp(-worse_by / temperature):
                x, e = y, ey
            temperature *= SA_COOLING
            if temperature < SA_MIN_TEMP:
                x = self.random_input()      # frozen: restart
                e = self.sa_energy(x, target)
                temperature = SA_START_TEMP
        return target in self.covered

    def sa_energy(self, args, target):
        """Execute `args` and score it for simulated annealing (lower is better).

            energy = approach_level * SA_APPROACH_WEIGHT + log(1 + branch distance)

        Why not the normal fitness? Its d / (d + 1) squeezes large distances
        together: d = 458 gives 0.9978 and d = 558 gives 0.9982. For hill
        climbing that is fine (it only asks "is it better?"), but SA asks "HOW
        MUCH worse?", sees almost no difference, and accepts nearly every bad
        move, turning into a random walk. The log scale compares relative
        changes instead (log 459 = 6.13 vs log 559 = 6.33), so moving away from
        the target is clearly worse no matter how far away we are.
        """
        r = self.evaluate(args)
        if is_covered(r, target):
            return 0.0
        approach, d = fitness_parts(r, target, self.branches)
        if d is None:
            return (approach + 1) * SA_APPROACH_WEIGHT
        return max(approach * SA_APPROACH_WEIGHT + math.log(d + 1), EPS)

    def search_tabu(self, target, budget):
        """Tabu search with restarts.  (--algorithm tabu)

        Like hill climbing it evaluates the whole neighbourhood, but it ALWAYS
        moves to the best neighbour, even if that neighbour is worse than the
        current input. To stop it from walking straight back, the last
        TABU_SIZE visited inputs are "tabu" (forbidden), unless a tabu input
        would be the best one seen so far (the "aspiration" rule).
        If the best fitness has not improved for TABU_PATIENCE moves, restart
        from a random input.
        """
        start = self.ex.evaluations
        x = self.start_point(target)
        f = self.fit(x, target)
        best_f, stale = f, 0
        tabu = collections.deque([tuple(x)], maxlen=TABU_SIZE)
        while self.ex.evaluations - start < budget and target not in self.covered:
            if f == 0:
                return True
            candidates = []
            for n in self.neighbours(x):
                nf = self.fit(n, target)
                if nf == 0:
                    return True
                if tuple(n) not in tabu or nf < best_f:
                    candidates.append((nf, n))
            if candidates:
                f, x = min(candidates, key=lambda c: c[0])
                tabu.append(tuple(x))
            if f < best_f:
                best_f, stale = f, 0
            else:
                stale += 1
            if not candidates or stale >= TABU_PATIENCE:
                x = self.random_input()      # stuck: restart
                f = self.fit(x, target)
                best_f, stale = f, 0
                tabu.clear()
        return target in self.covered

    def search_random(self, target, budget):
        """Baseline: random inputs until the target is covered.  (--algorithm random)

        Uses no fitness guidance at all; only useful to show that the guided
        algorithms do better.
        """
        start = self.ex.evaluations
        while self.ex.evaluations - start < budget and target not in self.covered:
            self.evaluate(self.random_input())
        return target in self.covered

    # ── choosing inputs to try ───────────────────────────────────────────────

    def start_point(self, target):
        """The best input seen so far for `target`, or a random one if there is none."""
        return list(self.best[target][1]) if target in self.best else self.random_input()

    def neighbours(self, x):
        """Yield every input that differs from `x` in one variable by +/- a step size.

        With STEPS = (1, 10, ..., 100000), each variable has 12 neighbours. Small
        steps fine-tune; big steps make the climb fast when the target value is
        far away (10 -> 42 takes 20, 30, 40, 41, 42 instead of 32 single steps).
        """
        for i in range(len(x)):
            for s in STEPS:
                for sign in (1, -1):
                    n = list(x)
                    n[i] += sign * s
                    yield n

    def random_neighbour(self, x):
        """One neighbour of `x` picked at random (used by simulated annealing)."""
        n = list(x)
        n[self.rng.randrange(len(x))] += self.rng.choice((1, -1)) * self.rng.choice(STEPS)
        return n

    def random_input(self):
        """Return a list of `n_args` random ints.

        mixing scales: small ranges make equalities between variables more likely.
        This matters for example3, whose hardest branches need degenerate lines,
        i.e. several coordinates being equal.
        """
        r = self.rng.choice(RANGES)
        return [self.rng.randint(-r, r) for _ in range(self.n_args)]

    # ── running one input ────────────────────────────────────────────────────

    def evaluate(self, args):
        """Run the target once with `args` and update the bookkeeping.

        This is the single place where the target is executed, so it also:
          - keeps the run as a test candidate if it covers something new
            (or if it is the first usable run, so every function gets a test)
          - adds the covered targets to `self.covered`
          - updates `self.best`, the best input seen so far for each uncovered
            target, which later becomes the starting point of the search
            algorithms (see start_point)
        """
        r = self.ex.run(self.func, args)
        if r.status != "abort":
            if not (r.taken <= self.covered) or not self.tests:
                self.tests.append(r)
            self.covered |= r.taken
        for t in self.targets:
            if t not in self.covered:
                f = fitness(r, t, self.branches)
                if t not in self.best or f < self.best[t][0]:
                    self.best[t] = (f, list(args))
        return r

    def fit(self, args, target):
        """Execute `args` and return its fitness for `target` (one fitness evaluation)."""
        return fitness(self.evaluate(args), target, self.branches)

    # ── choosing the final tests ─────────────────────────────────────────────

    def minimal_tests(self):
        """Greedy set cover over the collected tests, keeping discovery order.

        Repeatedly picks the test that covers the most still-uncovered targets,
        until no test adds anything. This keeps the generated file short while
        preserving the same coverage.
        """
        chosen, covered = [], set()
        pool = list(self.tests)
        while pool:
            best = max(pool, key=lambda r: len(r.taken - covered))
            if not (best.taken - covered) and chosen:
                break
            chosen.append(best)
            covered |= best.taken
            pool.remove(best)
        return sorted(chosen, key=self.tests.index)


# ═══════════════════════════════════════════════════════════════════════════
# 7. TEST FILE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def render_tests(module, suites):
    """Build the text of the PyTest file.

    `suites` is a list of (function name, [Result, ...]). Each result becomes

        def test_<func>_<i>():
            assert <module>.<func>(<args>) == <return value>

    or, if the run raised an exception,

        def test_<func>_<i>():
            with pytest.raises(<ExceptionType>):
                <module>.<func>(<args>)
    """
    lines, needs_pytest = [], False
    for func, results in suites:
        for i, r in enumerate(results, 1):
            call = f"{module}.{func}({', '.join(repr(a) for a in r.args)})"
            lines.append(f"def test_{func}_{i}():")
            if r.status == "exc":
                needs_pytest = True
                lines.append(f"    with pytest.raises({r.value.__name__}):")
                lines.append(f"        {call}")
            else:
                lines.append(f"    assert {call} == {r.value!r}")
            lines.append("")
    header = (["import pytest"] if needs_pytest else []) + [f"import {module}", "", ""]
    return "\n".join(header + lines)


if __name__ == "__main__":
    main()
