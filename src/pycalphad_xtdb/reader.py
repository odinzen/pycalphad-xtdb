"""Read XTDB files (Sundman et al., Calphad 90 (2025) 102849) into a pycalphad Database."""

import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field

from symengine.lib.symengine_wrapper import RealDouble
from tinydb import where

from pycalphad.core.utils import generate_symmetric_group
from pycalphad.io.tdb import (KNOWN_SUBLATTICE_SYMMETRY_RELATIONS, _molmass,
                              add_phase_symmetry_ordering_parameters)
from pycalphad.variables import Species

from . import expr as X
from .model import XTDBModel

# Structure factor p of the Inden-Hillert-Jarl function for each magnetic model Id, and
# the antiferromagnetic factor when the file gives none. IHJQX is the Xiong form, which
# pycalphad selects with an AFM factor of 0; p = 0.25 is the value behind the coefficients
# in the reference ModelDescriptions file.
MAGNETIC = {"IHJBCC": (0.40, -1.0), "IHJREST": (0.28, -3.0), "IHJQX": (0.25, 0.0)}
PERMUTATIONS = {"FCC4PERM": "symmetry_FCC_4SL", "BCC4PERM": "symmetry_BCC_4SL"}

# Model parameter identifiers (MPID) of the reference ModelDescriptions, mapped to the
# parameter types pycalphad's Model evaluates. A file's own ModelDescriptions extend this.
DEFAULT_MPID = {"LNTH": "THETA", "GD": "GD", "TC": "TC", "BMAGN": "BMAGN",
                "BMAG": "BMAGN", "CT": "TC", "NT": "NT", "L": "L", "G": "G"}

_PARAM_ID = re.compile(r"^\s*([A-Za-z0-9_]+)\s*\(\s*([^,]+?)\s*,\s*(.+?)\s*(?:;\s*(\d+)\s*)?\)\s*$")


@dataclass
class _Phase:
    name: str
    sites: list
    constituents: list
    hints: dict = field(default_factory=dict)
    disordered: str = None
    n_summed: int = 0
    synthesize_disordered: bool = False


def read_xtdb(dbf, fd):
    """Populate ``dbf`` from an XTDB file object (registered as the "xtdb" format)."""
    text = fd.read()
    root = _root(text)
    base = os.path.dirname(getattr(fd, "name", "") or "")
    extra = [_root(_read_append(base, path)) for path in _append_paths(root)]

    defaults = root.find("Defaults")
    d_low = _float(defaults.get("LowT")) if defaults is not None else None
    d_high = _float(defaults.get("HighT")) if defaults is not None else None

    mpid = dict(DEFAULT_MPID)
    afm = {}
    for tree in [root] + extra:
        for md in tree.iter("ModelDescriptions"):
            _read_model_descriptions(md, mpid, afm)

    _warn_global_models(root, defaults)
    _read_elements(dbf, root, defaults)
    R = _stated_gas_constant([root] + extra)
    if R is not None:
        dbf.symbols["R"] = RealDouble(R)

    gein, unreadable = {}, {}
    for tree in [root] + extra:
        for f in tree.iter("TPfun"):
            name = f.get("Id").upper()
            if name == "R":
                continue
            try:
                dbf.symbols[name] = _ranged(f, d_low, d_high, gein, R)
            except (ValueError, SyntaxError, TypeError) as err:
                # an unused broken function should not make the whole database unreadable
                unreadable[name] = f"{f.get('Expr')!r} ({type(err).__name__})"
                warnings.warn(f"TPfun {name} could not be read and is skipped: "
                              f"{unreadable[name]}")

    phases = {}
    for p in root.iter("Phase"):
        ph = _read_phase(p, afm)
        if ph is not None:
            phases[ph.name] = ph
    for ph in phases.values():
        dbf.add_structure_entry(ph.name, ph.name)
        dbf.add_phase(ph.name, ph.hints, ph.sites)
        dbf.add_phase_constituents(ph.name, ph.constituents)
    _link_disordered(dbf, phases)

    params = []
    for tree in [root] + extra:
        for el in tree.iter("Parameter"):
            p = _read_parameter(el, mpid, d_low, d_high, gein, R, phases)
            if p is not None:
                params.append(p)
    for p in _ternary_to_tdb_degrees(params):
        dbf.add_parameter(p["type"], p["phase"], p["constituents"], p["order"], p["expr"],
                          p["ref"], p.get("diffusing"), force_insert=False)

    for name, theta in gein.items():
        dbf.symbols[name] = X.gein_expression(theta, R if R is not None else X.v.R)
    _check_references(dbf, params, unreadable)

    dbf.process_parameter_queue()
    add_phase_symmetry_ordering_parameters(dbf)
    _permute_unordered_phases(dbf, phases)
    _warn_einstein_weighting(dbf)


def _warn_global_models(root, defaults):
    # tags directly under <XTDB> apply to the database; inside ModelDescriptions they only
    # describe what a model is
    for tag in root.findall("TernaryXpol"):
        if (tag.get("Phase") or "").strip():
            warnings.warn(f"TernaryXpol {tag.get('Xpol')} for {tag.get('Constituents')} in "
                          f"{tag.get('Phase')} is not applied: pycalphad extrapolates every "
                          f"ternary with the Muggianu method.")
    glob = (defaults.get("GlobalModel") or "") if defaults is not None else ""
    if root.findall("EEC") or "EEC" in glob.upper().split():
        warnings.warn("The database asks for the equal entropy criterion (EEC), which "
                      "pycalphad does not enforce; check that no solid is stable above "
                      "the liquid's entropy.")


def _check_references(dbf, params, unreadable):
    """Report functions that are used but missing; pycalphad would evaluate them as zero."""
    known = set(dbf.symbols) | {"T", "P"}
    users = [(f"TPfun {n}", e) for n, e in dbf.symbols.items()]
    users += [(f"{p['type']}({p['phase']},...)", p["expr"]) for p in params]
    missing = defaultdict(list)
    for who, e in users:
        for s in getattr(e, "free_symbols", ()):
            name = str(s)
            if name in known:
                continue
            if name in unreadable:
                raise ValueError(f"{who} uses TPfun {name}, which could not be read: "
                                 f"{unreadable[name]}")
            missing[name].append(who)
    for name, who in sorted(missing.items()):
        warnings.warn(f"{name} is used by {', '.join(sorted(set(who))[:3])} but not defined in "
                      f"the file; pycalphad will evaluate it as zero.")


def _root(text):
    try:
        root = ET.fromstring(text)
    except ET.ParseError as err:
        raise ValueError(f"not a well-formed XTDB (XML) file: {err}") from None
    if root.tag != "XTDB":
        raise ValueError(f"the root tag is <{root.tag}>, an XTDB file starts with <XTDB>")
    return root


def _append_paths(root):
    paths = []
    for tag in root.iter("AppendXTDB"):
        paths += [v for k, v in tag.attrib.items() if k != "Bibref" and v.strip()]
    return paths


def _read_append(base, path):
    full = os.path.join(base, path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"AppendXTDB file {path!r} not found next to the database")
    with open(full, encoding="utf-8") as f:
        return f.read()


def _float(s):
    return None if s is None or not str(s).strip() else float(str(s).replace("D", "E"))


def _read_model_descriptions(md, mpid, afm):
    for tag in md:
        ident = (tag.get("Id") or "").upper()
        if tag.tag == "Magnetic":
            if tag.get("Aff") and tag.get("Aff").strip():
                afm[ident] = float(tag.get("Aff"))
            for key, target in (("MPID1", "BMAGN"), ("MPID2", "TC"), ("MPID3", "NT")):
                if tag.get(key):
                    mpid[tag.get(key).strip().upper()] = target
        elif tag.tag == "Einstein" and tag.get("MPID1"):
            mpid[tag.get("MPID1").strip().upper()] = "THETA"
        elif tag.tag == "Liquid2state":
            if tag.get("MPID1"):
                mpid[tag.get("MPID1").strip().upper()] = "GD"
            if tag.get("MPID2"):
                mpid[tag.get("MPID2").strip().upper()] = "THETA"


def _read_elements(dbf, root, defaults):
    for name in ((defaults.get("Elements") if defaults is not None else None) or "").split():
        name = name.upper()
        dbf.elements.add(name)
        dbf.species.add(Species(name, {name: 1}))
    for el in root.iter("Element"):
        name = el.get("Id").upper()
        dbf.elements.add(name)
        mass = _float(el.get("Mass")) or _molmass.get(name, 0.0)
        dbf.refstates[name] = {"phase": (el.get("Refstate") or "").upper(), "mass": mass,
                               "H298": _float(el.get("H298")) or 0.0,
                               "S298": _float(el.get("S298")) or 0.0}
        dbf.species.add(Species(name, {name: 1}))
    elements = sorted(dbf.elements, key=len, reverse=True)
    for sp in root.iter("Species"):
        name = sp.get("Id").upper()
        if name in dbf.elements and Species(name) in dbf.species:
            continue
        constituents, charge = _stoichiometry(sp.get("Stoichiometry") or name, elements)
        dbf.species.add(Species(name, constituents, charge=charge))


def _stoichiometry(text, elements):
    """'AL2O3', 'FE/+2', 'O/-2', 'C1O2' -> ({element: amount}, charge)."""
    s = text.strip().upper().replace(" ", "")
    charge = 0
    m = re.search(r"/([+-])(\d*\.?\d*)$", s)
    if m:
        charge = (1 if m.group(1) == "+" else -1) * float(m.group(2) or 1)
        s = s[:m.start()]
    out, i = {}, 0
    while i < len(s):
        el = next((e for e in elements if s.startswith(e, i)), None)
        if el is None:
            raise ValueError(f"species stoichiometry {text!r} names an element not in the file")
        i += len(el)
        num = re.match(r"\d+(?:\.\d*)?(?:/\d+)?|\.\d+", s[i:])
        if num:
            a, _, b = num.group(0).partition("/")
            amount = float(a) / float(b) if b else float(a)
            i += len(num.group(0))
        else:
            amount = 1.0
        out[el] = out.get(el, 0.0) + amount
    return out, charge


def _stated_gas_constant(trees):
    for tree in trees:
        for f in tree.iter("TPfun"):
            if f.get("Id", "").upper() == "R" and len(f.findall("Trange")) == 0:
                try:
                    return float(f.get("Expr").strip().rstrip(";"))
                except ValueError:
                    return None
    return None


def _ranged(el, d_low, d_high, gein, R):
    low = _float(el.get("LowT"))
    low = d_low if low is None else low
    pieces = [(el.get("Expr"), el.get("HighT"))]
    pieces += [(t.get("Expr"), t.get("HighT")) for t in el.findall("Trange")]
    ranges = []
    for i, (text, high) in enumerate(pieces):
        hi = _float(high)
        if hi is None:
            hi = d_high
        ranges.append((X.parse(text, gein, R), hi))
    return X.piecewise(ranges, low)


def _read_phase(p, afm):
    name = p.get("Id").upper()
    config = (p.get("Configuration") or "CEF").upper()
    hints = {}
    if config == "I2SL":
        hints["ionic_liquid_2SL"] = True
    elif config != "CEF":
        warnings.warn(f"Phase {name} uses the {config} configuration, which this reader does "
                      f"not translate; the phase is left out of the database.")
        return None
    state = (p.get("State") or "").upper()
    if state == "L":
        hints["liquid"] = True
    elif state == "G":
        hints["gas"] = True

    subl = p.find("Sublattices")
    if subl is None:
        raise ValueError(f"Phase {name} has no Sublattices tag")
    sites = [float(x) for x in subl.get("Multiplicities").split()]
    constituents = [[] for _ in sites]
    for c in subl.findall("Constituents"):
        idx = int(c.get("Sublattice") or 1) - 1
        constituents[idx] = [x.upper() for x in c.get("List").split()]
    if int(subl.get("NumberOf") or len(sites)) != len(sites):
        raise ValueError(f"Phase {name}: NumberOf does not match the Multiplicities")
    ph = _Phase(name, sites, constituents, hints)

    amend = p.find("AmendPhase")
    if amend is None:
        return ph
    for model in (amend.get("Models") or "").upper().split():
        if model in MAGNETIC:
            p_factor, default_afm = MAGNETIC[model]
            hints["ihj_magnetic_structure_factor"] = p_factor
            hints["ihj_magnetic_afm_factor"] = afm.get(model, default_afm)
        elif model in PERMUTATIONS:
            hints[PERMUTATIONS[model]] = True
        elif model in ("GEIN", "LIQ2STATE", "EBEF"):
            pass  # carried by THETA/GD parameters and wildcards, nothing to set here
        else:
            warnings.warn(f"Phase {name}: model {model} is not evaluated by pycalphad; its "
                          f"contribution is missing from the Gibbs energy.")
    dis = amend.find("DisorderedPart")
    if dis is not None:
        if (dis.get("Subtract") or "").strip().upper() not in ("Y", "YES"):
            warnings.warn(f"Phase {name}: a DisorderedPart without Subtract=\"Y\" (Eq. 15 of "
                          f"the XTDB paper) has no counterpart in pycalphad, which always "
                          f"subtracts the ordered energy at the disordered state; the phase "
                          f"is left out of the database.")
            return None
        ph.n_summed = int(dis.get("Sum"))
        ph.disordered = (dis.get("Disordered") or "").strip().upper() or None
        if ph.disordered is None:
            ph.disordered = name + "_DIS"
            ph.synthesize_disordered = True
    return ph


def _link_disordered(dbf, phases):
    for ph in list(phases.values()):
        if ph.disordered is None:
            continue
        if ph.synthesize_disordered:
            n = ph.n_summed
            merged = sorted({c for subl in ph.constituents[:n] for c in subl})
            sites = [sum(ph.sites[:n])] + ph.sites[n:]
            constituents = [merged] + ph.constituents[n:]
            dis = _Phase(ph.disordered, sites, constituents, {})
            phases[dis.name] = dis
            dbf.add_structure_entry(dis.name, dis.name)
            dbf.add_phase(dis.name, dis.hints, dis.sites)
            dbf.add_phase_constituents(dis.name, dis.constituents)
        if ph.disordered not in dbf.phases:
            raise ValueError(f"Phase {ph.name}: disordered phase {ph.disordered} is not defined")
        hint = {"ordered_phase": ph.name, "disordered_phase": ph.disordered}
        dbf.phases[ph.name].model_hints.update(hint)
        dbf.phases[ph.disordered].model_hints.update(hint)


def _read_parameter(el, mpid, d_low, d_high, gein, R, phases):
    ident = el.get("Id")
    m = _PARAM_ID.match(ident or "")
    if m is None:
        raise ValueError(f"cannot read the parameter Id {ident!r}")
    kind, phase, const, order = m.group(1).upper(), m.group(2).upper(), m.group(3), m.group(4)
    phase, _, diffusing = phase.partition("&")  # TDB notation for mobility parameters
    if phase not in phases:
        return None  # its phase was dropped (with a warning) or is absent
    array = [[c.strip().upper() for c in subl.split(",")] for subl in const.split(":")]
    ph = phases[phase]
    if ph.synthesize_disordered and len(array) == 1 + len(ph.sites) - ph.n_summed:
        phase = ph.disordered
    expected = len(phases[phase].sites)
    # the ionic liquid takes neutral species with the cation sublattice left out, as in TDB
    ionic_neutral = phases[phase].hints.get("ionic_liquid_2SL") and len(array) == 1
    if len(array) != expected and not ionic_neutral:
        warnings.warn(f"{ident}: {len(array)} sublattices given but phase {phase} has "
                      f"{expected}; the parameter is skipped.")
        return None
    return {"type": mpid.get(kind, kind), "phase": phase,
            "constituents": [sorted(subl) for subl in array],
            "order": int(order or 0), "expr": _ranged(el, d_low, d_high, gein, R),
            "ref": el.get("Bibref"), "diffusing": diffusing or None}


def _ternary_to_tdb_degrees(params):
    """XTDB ternary degrees -> the TDB meaning pycalphad evaluates.

    XTDB (Eq. 23): y1 y2 y3 (0L + v1 1L + v2 2L + v3 3L), with 0L composition independent.
    TDB/pycalphad: a lone 0L is constant; with 1 or 2 present, y1 y2 y3 sum_k v_k (k-1)L.
    As v1 + v2 + v3 = 1 the XTDB form equals the TDB form with (k-1)L_tdb = 0L + kL.
    """
    groups = defaultdict(lambda: defaultdict(list))
    rest = []
    for p in params:
        sizes = [len(s) for s in p["constituents"]]
        if sorted(sizes)[-1] == 3 and sizes.count(3) == 1 and all(s == 1 for s in sizes if s != 3):
            key = (p["type"], p["phase"], p.get("diffusing"),
                   tuple(tuple(s) for s in p["constituents"]))
            groups[key][p["order"]].append(p)
        else:
            rest.append(p)
    for key, by_order in groups.items():
        if set(by_order) - {0, 1, 2, 3}:
            raise ValueError(f"ternary parameter {key} has a degree above 3")
        if any(len(ps) > 1 for ps in by_order.values()):
            warnings.warn(f"ternary parameter {key[0]}({key[1]},...) is given more than once "
                          f"for one degree; the values are added, as pycalphad does for TDB.")
        if set(by_order) <= {0}:
            rest += by_order[0]
            continue
        total = {k: sum((p["expr"] for p in ps), 0) for k, ps in by_order.items()}
        template = next(iter(by_order.values()))[0]
        for k in (1, 2, 3):
            ref = (by_order.get(k) or by_order.get(0) or [template])[0]["ref"]
            rest.append(dict(template, order=k - 1, expr=total.get(0, 0) + total.get(k, 0),
                             ref=ref))
    return rest


def _permute_unordered_phases(dbf, phases):
    """FCC4PERM/BCC4PERM on a phase without a disordered part.

    pycalphad generates the permutations only for ordered phases of an order-disorder
    pair (add_phase_symmetry_ordering_parameters); the XTDB permutation models apply to
    any phase that declares them.
    """
    for name, phase in dbf.phases.items():
        if phase.model_hints.get("ordered_phase") == name:
            continue
        for hint, symmetry in KNOWN_SUBLATTICE_SYMMETRY_RELATIONS.items():
            if not phase.model_hints.get(hint):
                continue
            for param in dbf.search(where("phase_name") == name):
                arr = param["constituent_array"]
                for new in set(generate_symmetric_group(arr, symmetry)) - {arr}:
                    new_param = dict(param, constituent_array=new,
                                     _generated_by_symmetry_option=True)
                    dbf._parameters.insert(new_param)


def _warn_einstein_weighting(dbf):
    multi = set()
    for param in dbf.search(where("parameter_type") == "THETA"):
        phase = dbf.phases[param["phase_name"]]
        atoms = sum(n for n, subl in zip(phase.sublattices, param["constituent_array"])
                    if not all(c.name in ("VA", "*") for c in subl))
        if abs(atoms - 1) > 1e-12:
            multi.add(phase.name)
    if multi:
        warnings.warn(
            "Einstein (LNTH) parameters on endmembers with more than one atom per formula "
            f"unit in {sorted(multi)}. Calculate with model=pycalphad_xtdb.XTDBModel; "
            "pycalphad's default Model gives these endmembers one Einstein oscillator per "
            "formula unit instead of one per atom (XTDB paper, Eq. 12).", stacklevel=3)


__all__ = ["read_xtdb", "XTDBModel"]
