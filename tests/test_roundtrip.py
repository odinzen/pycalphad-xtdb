"""Every TDB bundled with pycalphad, written as XTDB and read back, gives the same energies."""

import pathlib
import warnings

import numpy as np
import pytest

import pycalphad.tests.databases as bundled
import pycalphad_xtdb
from pycalphad import Database, Model, calculate

TDBS = sorted(pathlib.Path(bundled.__file__).parent.glob("*.tdb"))
TEMPERATURES = [300.0, 1000.0, 2000.0]


@pytest.mark.parametrize("path", TDBS, ids=lambda p: p.name)
def test_tdb_to_xtdb_roundtrip(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db1 = Database(str(path))
        db2 = Database.from_string(pycalphad_xtdb.xtdb_text(db1), fmt="xtdb")
        comps = sorted(db1.elements - {"/-"})
        compared = 0
        for phase in sorted(db1.phases):
            try:
                m1 = Model(db1, comps, phase)
            except Exception:
                continue  # pycalphad cannot build this phase from the TDB either
            m2 = Model(db2, comps, phase)
            sample = calculate(db1, comps, phase, T=TEMPERATURES, P=101325, N=1, pdens=20,
                               model={phase: m1}, output="GM")
            y = sample.Y.values.reshape(-1, sample.Y.shape[-1])
            y = y[~np.isnan(y).any(axis=1)]
            # pycalphad samples endmembers and edges first; draw from the whole set so
            # interior points (where ternary and reciprocal terms act) are compared too
            y = np.unique(y, axis=0)
            y = y[np.random.default_rng(0).permutation(len(y))[:60]]
            g1, g2 = (calculate(db, comps, phase, T=TEMPERATURES, P=101325, N=1, points=y,
                                model={phase: m}, output="GM").GM.values
                      for db, m in ((db1, m1), (db2, m2)))
            np.testing.assert_allclose(g2, g1, rtol=1e-11, atol=1e-6, err_msg=phase)
            compared += 1
    assert compared > 0
