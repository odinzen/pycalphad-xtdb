"""Write a pycalphad Database as an XTDB file."""

import datetime
import warnings
from collections import defaultdict
from xml.sax.saxutils import quoteattr

from . import __version__
from . import expr as X

_TYPE_TO_MPID = {"THETA": "LNTH"}
_PERM_MODELS = {"symmetry_FCC_4SL": "FCC4PERM", "symmetry_BCC_4SL": "BCC4PERM"}


def write_xtdb(dbf, fd, signature="pycalphad-xtdb", date=None):
    """Write ``dbf`` to the file object ``fd`` (registered as the "xtdb" format).

    Ternary interaction degrees are converted to the XTDB meaning (Eq. 23 of the paper),
    disordered parts are written as their own phases named in the DisorderedPart tag,
    and parameters pycalphad generated from a permutation model are left out.
    """
    fd.write(xtdb_text(dbf, signature=signature, date=date))


def xtdb_text(dbf, signature="pycalphad-xtdb", date=None):
    date = date or datetime.date.today().strftime("%Y/%m/%d")
    gein = {name for name in dbf.symbols if _is_gein_helper(dbf, name)}
    lines = [f'<XTDB Version="0.1.5" Software={quoteattr("pycalphad-xtdb " + __version__)} '
             f'Date={quoteattr(date)} Signature={quoteattr(signature)}>']

    special = [e for e in ("VA", "/-") if e in dbf.elements]
    if special:
        lines.append(f'<Defaults Elements={quoteattr(" ".join(special))} />')
    for el in sorted(dbf.elements - set(special)):
        ref = dbf.refstates.get(el, {})
        lines.append(f'<Element Id={quoteattr(el)} Refstate={quoteattr(ref.get("phase", ""))} '
                     f'Mass="{ref.get("mass", 0.0)}" H298="{ref.get("H298", 0.0)}" '
                     f'S298="{ref.get("S298", 0.0)}" />')
    for sp in sorted(dbf.species, key=lambda s: s.name):
        lines.append(f'<Species Id={quoteattr(sp.name)} '
                     f'Stoichiometry={quoteattr(_stoichiometry(sp))} />')

    params = [p for p in dbf._parameters.all() if not p.get("_generated_by_symmetry_option")]
    for name in sorted(dbf.phases):
        lines += _phase(dbf, dbf.phases[name], params)

    # R first, so a reader that looks for the gas constant finds it before it is used
    for name in sorted(dbf.symbols, key=lambda n: (n != "R", n)):
        if name not in gein:
            lines += _ranged("TPfun", f"Id={quoteattr(name)}", dbf.symbols[name], gein)

    refs = set()
    for p in _ternary_to_xtdb_degrees(params):
        mpid = _TYPE_TO_MPID.get(p["parameter_type"], p["parameter_type"])
        if mpid == "TC" and dbf.phases[p["phase_name"]].model_hints.get(
                "ihj_magnetic_afm_factor") == 0:
            mpid = "CT"
        const = ":".join(",".join(c.name for c in subl) for subl in p["constituent_array"])
        phase_field = p["phase_name"]
        diffusing = getattr(p.get("diffusing_species"), "name", None)
        if diffusing:
            # mobility parameters keep the TDB notation PHASE&SPECIES
            phase_field += "&" + diffusing
        ref = p.get("reference") or "UNKNOWN"
        refs.add(ref)
        ident = f"{mpid}({phase_field},{const};{p['parameter_order']})"
        lines += _ranged("Parameter", f"Id={quoteattr(ident)}", p["parameter"], gein,
                         tail=f" Bibref={quoteattr(ref)}")

    lines.append("<Bibliography>")
    for ref in sorted(refs):
        text = "No reference given in the source database" if ref == "UNKNOWN" else ref
        lines.append(f'  <Bibitem Id={quoteattr(ref)} Text={quoteattr(text)} />')
    lines.append("</Bibliography>")
    lines.append("</XTDB>")
    return "\n".join(lines) + "\n"


def _is_gein_helper(dbf, name):
    if not name.startswith(X.GEIN_PREFIX):
        return False
    try:
        theta = X.gein_theta(name)
    except ValueError:
        return False
    from .model import gas_constant
    expected = X.gein_expression(theta, gas_constant(dbf))
    actual = dbf.symbols[name]
    return all(abs(float(actual.subs({X.v.T: t})) - float(expected.subs({X.v.T: t}))) < 1e-9
               for t in (50.0, 300.0, 2000.0))


def _stoichiometry(sp):
    if not sp.constituents:
        return sp.name
    out = "".join(f"{el}{_amount(n)}" for el, n in sorted(sp.constituents.items()))
    if sp.charge:
        sign = "+" if sp.charge > 0 else "-"
        out += f"/{sign}{_amount(abs(sp.charge))}"
    return out


def _amount(n):
    n = float(n)
    return str(int(n)) if n == int(n) else repr(n)


def _phase(dbf, phase, params):
    hints = phase.model_hints
    attrs = [f"Id={quoteattr(phase.name)}",
             'Configuration="I2SL"' if hints.get("ionic_liquid_2SL") else 'Configuration="CEF"']
    if hints.get("liquid"):
        attrs.append('State="L"')
    elif hints.get("gas"):
        attrs.append('State="G"')
    else:
        attrs.append('State="S"')
    out = [f"<Phase {' '.join(attrs)} >"]
    mult = " ".join(_amount(s) for s in phase.sublattices)
    out.append(f'  <Sublattices NumberOf="{len(phase.sublattices)}" Multiplicities="{mult}" >')
    for i, subl in enumerate(phase.constituents):
        names = " ".join(sorted(c.name for c in subl))
        out.append(f'    <Constituents Sublattice="{i + 1}" List={quoteattr(names)} />')
    out.append("  </Sublattices>")

    models = []
    if "ihj_magnetic_afm_factor" in hints:
        models.append(_magnetic_model(phase))
    types = {p["parameter_type"] for p in params if p["phase_name"] == phase.name}
    if "THETA" in types:
        models.append("GEIN")
    if "GD" in types:
        models.append("LIQ2STATE")
    models += [m for hint, m in _PERM_MODELS.items() if hints.get(hint)]
    dis = None
    if hints.get("ordered_phase") == phase.name and hints.get("disordered_phase"):
        dis = dbf.phases[hints["disordered_phase"]]
    if models or dis is not None:
        tag = f'  <AmendPhase Models={quoteattr(" ".join(models))}' if models else "  <AmendPhase"
        if dis is None:
            out.append(tag + " />")
        else:
            out.append(tag + " >")
            n_summed = len(phase.sublattices) - (len(dis.sublattices) - 1)
            out.append(f'    <DisorderedPart Disordered={quoteattr(dis.name)} Sum="{n_summed}" '
                       f'Subtract="Y" />')
            out.append("  </AmendPhase>")
    out.append("</Phase>")
    return out


def _magnetic_model(phase):
    afm = phase.model_hints["ihj_magnetic_afm_factor"]
    p = phase.model_hints["ihj_magnetic_structure_factor"]
    known = {(-1.0, 0.4): "IHJBCC", (-3.0, 0.28): "IHJREST", (0.0, 0.25): "IHJQX"}
    for (a, s), name in known.items():
        if abs(afm - a) < 1e-9 and abs(p - s) < 1e-9:
            return name
    if afm == 0:
        warnings.warn(f"Phase {phase.name}: Xiong magnetic model with structure factor {p}; "
                      f"XTDB's IHJQX carries 0.25, so the written file differs for this phase.")
        return "IHJQX"
    raise ValueError(f"Phase {phase.name}: magnetic model with AFM factor {afm} and structure "
                     f"factor {p} has no XTDB model Id")


def _ternary_to_xtdb_degrees(params):
    """Inverse of the reader's conversion: TDB (k-1)L becomes XTDB kL, no 0L."""
    groups = defaultdict(lambda: defaultdict(list))  # duplicates kept: pycalphad sums them
    out = []
    for p in params:
        sizes = [len(s) for s in p["constituent_array"]]
        if max(sizes) == 3 and sizes.count(3) == 1 and all(s == 1 for s in sizes if s != 3):
            key = (p["parameter_type"], p["phase_name"], p.get("diffusing_species"),
                   p["constituent_array"])
            groups[key][p["parameter_order"]].append(p)
        else:
            out.append(p)
    for by_order in groups.values():
        if set(by_order) == {0}:
            out += by_order[0]
            continue
        for k in (0, 1, 2):
            out += [dict(p, parameter_order=k + 1) for p in by_order.get(k, [])]
    return out


def _ranged(tag, ident, expr, gein, tail=""):
    low, pieces = X.split_piecewise(expr)
    low_attr = f' LowT="{_amount(low)}"' if low is not None else ""
    first, hi = pieces[0]
    hi_attr = f' HighT="{_amount(hi)}"' if hi is not None else ""
    head = f'<{tag} {ident}{low_attr} Expr="{X.to_text(first, gein)};"{hi_attr}{tail}'
    if len(pieces) == 1:
        return [head + " />"]
    out = [head + " >"]
    for e, hi in pieces[1:]:
        hi_attr = f' HighT="{_amount(hi)}"' if hi is not None else ""
        out.append(f'  <Trange Expr="{X.to_text(e, gein)};"{hi_attr} />')
    out.append(f"</{tag}>")
    return out
