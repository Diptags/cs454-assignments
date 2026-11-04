"""Small helper functions used by sbst.py.

Everything here is a plain function with no state. They are grouped by the
section of sbst.py that uses them:

  A. AST builders ........ rt_attr, rt_call, thunk, terminates   (used by Instrumenter)
  B. Distance helpers .... empty_distance                         (used by Runtime)
"""
import ast

# Shared constants (also imported by sbst.py)
K = 1                    # constant added to branch distances of strict comparisons
RT = "__sbst_rt__"       # name of the runtime recorder inside the instrumented module


# ═══════════════════════════════════════════════════════════════════════════
# A. AST BUILDERS
#
# rt_attr, rt_call and thunk do not run anything: they BUILD pieces of code as
# AST nodes, which the Instrumenter puts into the target program.
#   rt_attr("distance_compare")
#       →  __sbst_rt__.distance_compare
#   rt_call("distance_compare", "==", x, 42)
#       →  __sbst_rt__.distance_compare('==', x, 42)
#   thunk(<the call above>)
#       →  lambda: __sbst_rt__.distance_compare('==', x, 42)
# ═══════════════════════════════════════════════════════════════════════════

def rt_attr(attr):
    """Build the AST for `__sbst_rt__.<attr>` (an attribute of the runtime recorder)."""
    return ast.Attribute(value=ast.Name(id=RT, ctx=ast.Load()), attr=attr, ctx=ast.Load())


def rt_call(attr, *args):
    """Build the AST for the call `__sbst_rt__.<attr>(*args)`."""
    return ast.Call(func=rt_attr(attr), args=list(args), keywords=[])


def thunk(expr):
    """Build the AST for `lambda: <expr>`.

    Wrapping an operand in a lambda delays its evaluation, which lets
    `distance_and` and `distance_or` keep Python's short-circuit behaviour.
    """
    no_args = ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[],
                            kw_defaults=[], kwarg=None, defaults=[])
    return ast.Lambda(args=no_args, body=expr)


def terminates(stmts):
    """True if control never falls through the end of this statement list.

    Used to spot "early exit" patterns such as

        if u1t == 0:
            return True      # body always leaves
        if u2t == 0:         # only reached when u1t == 0 was False
            ...

    A list terminates if it contains a return / raise / break / continue, or an
    if-else whose two sides both terminate.
    """
    for s in stmts:
        if isinstance(s, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            return True
        if isinstance(s, ast.If) and s.orelse and terminates(s.body) and terminates(s.orelse):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# B. DISTANCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def empty_distance(iterable):
    """Distance for an empty iterable to become non-empty (only `range` gives a gradient).

    Example: range(0, -5) is empty; start - stop + K = 0 - (-5) + 1 = 6, so the
    search learns that making the stop value bigger helps.
    """
    if isinstance(iterable, range):
        if iterable.step > 0:
            return iterable.start - iterable.stop + K
        return iterable.stop - iterable.start + K
    return K
