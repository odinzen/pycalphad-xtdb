"""Al-C third-generation database: values, invariants and heat capacity."""

import json
import pathlib
import warnings

import numpy as np
import pytest

import pycalphad_xtdb
from pycalphad import Database, Model, calculate, equilibrium, variables as v

HERE = pathlib.Path(__file__).parent
EXAMPLES = HERE.parent / "examples"
R = 8.31451

POINTS = {  # site fractions in pycalphad's sorted constituent order
    ("GRAPHITE", "C"): (["C"], [1.0]),
    ("DIAMOND", "C"): (["C"], [1.0]),
    ("AL4C3", "AL:C"): (["AL", "C"], [1.0, 1.0]),
    ("FCC_A1", "AL:VA"): (["AL", "C", "VA"], [1.0, 1e-14, 1 - 1e-14]),
    ("FCC_A1", "AL:C"): (["AL", "C", "VA"], [1.0, 1 - 1e-14, 1e-14]),
    ("BCC_A2", "AL:VA"): (["AL", "C", "VA"], [1.0, 1e-14, 1 - 1e-14]),
    ("BCC_A2", "AL:C"): (["AL", "C", "VA"], [1.0, 1 - 1e-14, 1e-14]),
    ("HCP_A3", "AL:VA"): (["AL", "C", "VA"], [1.0, 1e-14, 1 - 1e-14]),
    ("HCP_A3", "AL:C"): (["AL", "C", "VA"], [1.0, 1 - 1e-14, 1e-14]),
    ("LIQUID", "AL"): (["AL"], [1.0]),
    ("LIQUID", "C"): (["C"], [1.0]),
}


def _db(text=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if text is not None:
            return Database.from_string(text, fmt="xtdb")
        return Database(str(EXAMPLES / "AlC.xtdb"))


def _gm(db, phase, comps, y, T, model=pycalphad_xtdb.XTDBModel, output="GM"):
    res = calculate(db, comps, phase, T=T, P=101325, N=1, points=np.array([y]),
                    model=model, output=output)
    return float(getattr(res, output).values.squeeze())


def _reference():
    return json.loads((HERE / "data" / "alc_endmembers_reference.json").read_text())["rows"]


@pytest.mark.parametrize("row", _reference(), ids=lambda r: f"{r['phase']}-{r['endmember']}-{r['T']}")
def test_endmembers_match_independent_evaluator(row):
    comps, y = POINTS[(row["phase"], row["endmember"])]
    g = _gm(_db(), row["phase"], comps, y, row["T"])
    assert g == pytest.approx(row["G_per_mole_atoms"], abs=1e-6, rel=0)


def test_planted_parameter_error_is_caught():
    # +350 J on the Al4C3 formula unit is +50 J per mole of atoms; the check above must see it
    text = (EXAMPLES / "AlC.xtdb").read_text().replace('Expr="-277339-', 'Expr="-276989-')
    row = next(r for r in _reference() if r["phase"] == "AL4C3" and r["T"] == 300.0)
    g = _gm(_db(text), "AL4C3", ["AL", "C"], [1.0, 1.0], 300.0)
    assert g - row["G_per_mole_atoms"] == pytest.approx(50.0, abs=1e-6)


def test_default_model_misweights_compound_einstein_term():
    db = _db()
    g_x = _gm(db, "AL4C3", ["AL", "C"], [1.0, 1.0], 300.0)
    g_d = _gm(db, "AL4C3", ["AL", "C"], [1.0, 1.0], 300.0, model=Model)
    assert abs(g_x - g_d) > 1000.0
    # one-atom endmembers agree up to the gas constant (8.31451 here, 8.3145 in pycalphad)
    g_x = _gm(db, "GRAPHITE", ["C"], [1.0], 300.0)
    g_d = _gm(db, "GRAPHITE", ["C"], [1.0], 300.0, model=Model)
    assert abs(g_x - g_d) < 0.05


def test_reading_warns_about_compound_einstein_terms():
    with pytest.warns(UserWarning, match="XTDBModel"):
        Database(str(EXAMPLES / "AlC.xtdb"))


def test_al4c3_heat_capacity_is_physical():
    db = _db()
    cp = [_gm(db, "AL4C3", ["AL", "C"], [1.0, 1.0], T, output="CPM") / R
          for T in (50.0, 300.0, 1000.0, 2000.0)]
    assert all(c > 0 for c in cp)
    assert cp[0] < cp[1] < cp[2] < cp[3]
    assert 2.9 < cp[2] < 3.2  # per mole of atoms, near Dulong-Petit
    cp_default = _gm(db, "AL4C3", ["AL", "C"], [1.0, 1.0], 300.0, model=Model, output="CPM")
    assert cp_default < 0  # the failure XTDBModel exists to prevent


def _stable(db, T, xc=None):
    comps, conds = ["AL", "C", "VA"], {v.T: T, v.P: 101325, v.N: 1}
    if xc is None:
        comps = ["AL", "VA"]
    else:
        conds[v.X("C")] = xc
    eq = equilibrium(db, comps, list(db.phases), conds, model=pycalphad_xtdb.XTDBModel)
    return {str(p) for p in eq.Phase.values.ravel() if str(p)}


def test_aluminium_melting_point():
    db = _db()
    assert _stable(db, 933.4) == {"FCC_A1"}
    assert _stable(db, 933.55) == {"LIQUID"}


def test_al4c3_peritectic():
    # He et al. (2021) model; measured decomposition 2156 C = 2429 K
    db = _db()
    assert _stable(db, 2415.0, 0.5) == {"AL4C3", "GRAPHITE"}
    assert _stable(db, 2435.0, 0.5) == {"GRAPHITE", "LIQUID"}


def test_write_then_read_keeps_gein_and_values():
    db = _db()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # XTDB-origin LNTH keeps its meaning, so no warning
        text = pycalphad_xtdb.xtdb_text(db)
    assert "GEIN(1953.3)" in text and "LNTH(AL4C3,AL:C;0)" in text
    assert "GD(LIQUID,AL;0)" in text and "LIQ2STATE" in text
    assert 'Id="R" Expr="+8.31451;"' in text
    db2 = _db(text)
    for (phase, cons), (comps, y) in POINTS.items():
        for T in (10.0, 1500.0):
            assert _gm(db2, phase, comps, y, T) == pytest.approx(_gm(db, phase, comps, y, T),
                                                                 abs=1e-6, rel=0)
