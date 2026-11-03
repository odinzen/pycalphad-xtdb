"""Conversion between XTDB Expr strings and the symengine expressions pycalphad stores."""

import math
import re

from symengine import And, Piecewise, Symbol, exp, log
from symengine.lib.symengine_wrapper import Add, Mul, Pow, Integer, RealDouble, Rational

from pycalphad import variables as v
from pycalphad.io.tdb import _sympify_string

_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?"
_GEIN_CALL = re.compile(r"GEIN\s*\(\s*(" + _NUMBER + r")\s*\)", re.IGNORECASE)
_R_NAME = re.compile(r"(?<![A-Za-z0-9_#])R#?(?![A-Za-z0-9_(])")
GEIN_PREFIX = "GEIN_"


def gein_symbol_name(theta):
    """Name of the helper symbol holding GEIN(theta); theta must be a plain number."""
    text = repr(float(theta)).replace("-", "M").replace("+", "").replace(".", "P")
    return GEIN_PREFIX + text.upper()


def gein_expression(theta, R=v.R):
    """One Einstein oscillator per mole of atoms, Eq. (11) of Sundman et al. (2025)."""
    theta = float(theta)
    return 1.5 * R * theta + 3 * R * v.T * log(1 - exp(-theta / v.T))


def parse(text, gein_thetas, R=None):
    """XTDB Expr -> symengine expression (no temperature bounds).

    GEIN(theta) calls become helper symbols whose names are collected in ``gein_thetas``
    (name -> theta), so the caller can define them once in ``Database.symbols``. When the
    file states its own gas constant, ``R`` is that value and replaces the bare name R.
    """
    s = text.strip().rstrip(";").strip()
    if not s:
        raise ValueError("empty Expr")
    if re.search(r"GEIN\s*\(", s, re.IGNORECASE) and not _GEIN_CALL.search(s):
        raise ValueError(f"GEIN needs a numeric Einstein temperature as argument: {text!r}")

    def _gein(m):
        theta = float(m.group(1).replace("D", "E").replace("d", "e"))
        name = gein_symbol_name(theta)
        gein_thetas[name] = theta
        return name

    s = _GEIN_CALL.sub(_gein, s)
    if R is not None:
        s = _R_NAME.sub(f"({R!r})", s)
    # TDB-style Fortran exponents (1.0D-5) are not Python syntax
    s = re.sub(r"(\d)[Dd]([+-]?\d)", r"\1E\2", s)
    return _sympify_string(s)


def piecewise(ranges, low):
    """Assemble [(expr, high_T), ...] the way pycalphad's TDB reader builds a FUNCTION."""
    if low is None and len(ranges) == 1 and ranges[0][1] is None:
        return ranges[0][0]
    pairs = []
    lo = low
    for expr, hi in ranges:
        conds = []
        if lo is not None:
            conds.append(lo <= v.T)
        if hi is not None:
            conds.append(v.T < hi)
        pairs.append((expr, And(*conds) if conds else True))
        lo = hi
    pairs.append((0, True))
    return Piecewise(*pairs)


def split_piecewise(expr):
    """Inverse of :func:`piecewise`: (low_T, [(expr, high_T), ...])."""
    if not isinstance(expr, Piecewise):
        return None, [(expr, None)]
    args = expr.args
    pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
    if pairs and pairs[-1][1] == True and pairs[-1][0] == 0:  # noqa: E712 (symengine true)
        pairs = pairs[:-1]
    low, out = None, []
    for i, (e, cond) in enumerate(pairs):
        lo, hi = _bounds(cond)
        if i == 0:
            low = lo
        out.append((e, hi))
    return low, out


def _bounds(cond):
    lo = hi = None
    parts = cond.args if type(cond).__name__ == "And" else (cond,)
    for rel in parts:
        name = type(rel).__name__
        a, b = rel.args
        if name == "LessThan" and b == v.T:          # lo <= T
            lo = float(a)
        elif name == "StrictLessThan" and a == v.T:  # T < hi
            hi = float(b)
        elif name == "LessThan" and a == v.T:
            hi = float(b)
        elif name == "StrictLessThan" and b == v.T:
            lo = float(a)
        elif rel == True:  # noqa: E712
            continue
        else:
            raise ValueError(f"unsupported temperature condition {rel}")
    return lo, hi


def _num(x):
    f = float(x)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def _is_e(x):
    try:
        return abs(float(x) - math.e) < 1e-12
    except (TypeError, RuntimeError):
        return False


def to_text(expr, gein_names=()):
    """symengine expression -> XTDB/TDB Expr text (without the closing ';')."""
    out = _fmt(expr, set(gein_names))
    return out if out.startswith(("+", "-")) else "+" + out


def _fmt(e, gein):
    if isinstance(e, (Integer, RealDouble, Rational)) or e.is_Number:
        return _num(e)
    if isinstance(e, Symbol) or e.is_Symbol:
        name = str(e)
        if name in gein:
            return "GEIN(" + _num(gein_theta(name)) + ")"
        return name
    if isinstance(e, Add):
        terms = sorted(e.args, key=_term_order)
        out = ""
        for t in terms:
            s = _fmt(t, gein)
            out += s if s.startswith("-") else "+" + s
        return out
    if isinstance(e, Mul):
        coeff, factors = 1.0, []
        for f in e.args:
            if f.is_Number:
                coeff *= float(f)
            else:
                factors.append(f)
        body = "*".join(_factor(f, gein) for f in factors)
        if not factors:
            return _num(coeff)
        if coeff == 1:
            return body
        if coeff == -1:
            return "-" + body
        return _num(coeff) + "*" + body
    if isinstance(e, Pow):
        base, ex = e.args
        if _is_e(base) or str(base) == "E":
            return "EXP(" + _fmt(ex, gein) + ")"
        b = _factor(base, gein)
        if ex.is_Number and float(ex) == int(float(ex)):
            n = int(float(ex))
            return f"{b}**{n}" if n >= 0 else f"{b}**({n})"
        return f"{b}**({_fmt(ex, gein)})"
    if type(e).__name__ == "log":
        return "LN(" + _fmt(e.args[0], gein) + ")"
    raise ValueError(f"cannot write {type(e).__name__} in an XTDB Expr: {e}")


def _factor(f, gein):
    s = _fmt(f, gein)
    if isinstance(f, Add) or s.startswith("-"):
        return "(" + s + ")"
    return s


def _term_order(t):
    # constant first, then the rest in a stable textual order
    return (0 if t.is_Number else 1, str(t))


def gein_theta(name):
    """Recover theta from a helper symbol name made by :func:`gein_symbol_name`."""
    body = name[len(GEIN_PREFIX):].replace("M", "-").replace("P", ".")
    return float(body)
