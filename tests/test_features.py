"""Format features checked against the formulas of the XTDB paper, and loud failures."""

import itertools
import math
import warnings

import numpy as np
import pytest

import pycalphad_xtdb  # noqa: F401  (registers the format)
from pycalphad_xtdb.reader import _supports_never_disorder
from pycalphad import Database, Model, calculate, variables as v

RT = float(v.R) * 1000.0


def xtdb(body, elements=("AG", "AU", "CU")):
    els = "\n".join(f'<Element Id="{e}" Refstate="FCC_A1" Mass="1" H298="0" S298="0" />'
                    for e in elements)
    sps = "\n".join(f'<Species Id="{e}" Stoichiometry="{e}" />' for e in elements)
    return (f'<XTDB Version="0.1.5" Software="test" Date="2026/11/03" Signature="test">\n'
            f'<Defaults Elements="VA /-" />\n{els}\n<Species Id="VA" Stoichiometry="VA" />\n'
            f'{sps}\n{body}\n</XTDB>\n')


def read(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Database.from_string(text, fmt="xtdb")


def gm(db, phase, comps, y, T=1000.0, model=Model):
    res = calculate(db, comps, phase, T=T, P=101325, N=1, points=np.atleast_2d(y),
                    model=model, output="GM")
    return res.GM.values.squeeze()


LIQUID = """<Phase Id="LIQUID" Configuration="CEF" State="L" >
  <Sublattices NumberOf="1" Multiplicities="1" >
    <Constituents Sublattice="1" List="AG AU CU" />
  </Sublattices>
</Phase>"""


def ideal(y):
    return RT * sum(x * math.log(x) for x in y)


def test_ternary_degrees_follow_eq23():
    # XTDB: y1 y2 y3 (0L + v1 1L + v2 2L + v3 3L); in one sublattice v_i = y_i
    db = read(xtdb(LIQUID + """
<Parameter Id="G(LIQUID,AG,AU,CU;0)" Expr="1000;" Bibref="t" />
<Parameter Id="G(LIQUID,AG,AU,CU;1)" Expr="2000;" Bibref="t" />
<Parameter Id="G(LIQUID,AG,AU,CU;3)" Expr="-500;" Bibref="t" />"""))
    for y in ([0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [1 / 3, 1 / 3, 1 / 3]):
        expected = ideal(y) + y[0] * y[1] * y[2] * (1000 + y[0] * 2000 + y[2] * -500)
        assert gm(db, "LIQUID", ["AG", "AU", "CU"], y) == pytest.approx(expected, abs=1e-8)


def test_lone_ternary_zero_degree_is_constant():
    db = read(xtdb(LIQUID + '<Parameter Id="L(LIQUID,AG,AU,CU;0)" Expr="3000;" Bibref="t" />'))
    y = [0.2, 0.3, 0.5]
    expected = ideal(y) + y[0] * y[1] * y[2] * 3000
    assert gm(db, "LIQUID", ["AG", "AU", "CU"], y) == pytest.approx(expected, abs=1e-8)


def test_ternary_degrees_survive_write_and_read():
    db = read(xtdb(LIQUID + """
<Parameter Id="G(LIQUID,AG,AU,CU;0)" Expr="1000;" Bibref="t" />
<Parameter Id="G(LIQUID,AG,AU,CU;2)" Expr="700;" Bibref="t" />"""))
    text = pycalphad_xtdb.xtdb_text(db)
    assert "G(LIQUID,AG,AU,CU;0)" not in text  # written as 1L..3L with 0L folded in
    db2 = read(text)
    y = [0.2, 0.3, 0.5]
    assert gm(db2, "LIQUID", ["AG", "AU", "CU"], y) == pytest.approx(
        gm(db, "LIQUID", ["AG", "AU", "CU"], y), abs=1e-8)


def test_wildcard_is_independent_of_the_other_sublattice():
    body = """<Phase Id="TWO" Configuration="CEF" State="S" >
  <Sublattices NumberOf="2" Multiplicities="1 1" >
    <Constituents Sublattice="1" List="AG AU" />
    <Constituents Sublattice="2" List="CU VA" />
  </Sublattices>
</Phase>
<Parameter Id="G(TWO,AG,AU:*;0)" Expr="5000;" Bibref="t" />
<Parameter Id="G(TWO,AG,AU:CU;0)" Expr="-7000;" Bibref="t" />"""
    db = read(xtdb(body))
    yag, ycu = 0.3, 0.4
    y = [yag, 1 - yag, ycu, 1 - ycu]
    expected_xs = yag * (1 - yag) * (5000 + ycu * -7000)  # Eq. (17)
    conf = RT * (yag * math.log(yag) + (1 - yag) * math.log(1 - yag)
                 + ycu * math.log(ycu) + (1 - ycu) * math.log(1 - ycu))
    atoms = 1 + ycu
    assert gm(db, "TWO", ["AG", "AU", "CU", "VA"], y) == pytest.approx(
        (conf + expected_xs) / atoms, abs=1e-8)


def _ordered(perm, synthesized):
    ordered = """<Phase Id="FCC_L12" Configuration="CEF" State="S" >
  <Sublattices NumberOf="4" Multiplicities="0.25 0.25 0.25 0.25" >
""" + "".join(f'    <Constituents Sublattice="{i}" List="AG CU" />\n' for i in (1, 2, 3, 4)) + \
        "  </Sublattices>\n"
    models = ' Models="FCC4PERM"' if perm else ""
    dis_attr = "" if synthesized else ' Disordered="FCC_A1"'
    ordered += f'  <AmendPhase{models} >\n    <DisorderedPart{dis_attr} Sum="4" Subtract="Y" />\n'
    ordered += "  </AmendPhase>\n</Phase>\n"
    dis_phase = "FCC_L12" if synthesized else "FCC_A1"
    body = ordered
    if not synthesized:
        body += """<Phase Id="FCC_A1" Configuration="CEF" State="S" >
  <Sublattices NumberOf="1" Multiplicities="1" >
    <Constituents Sublattice="1" List="AG CU" />
  </Sublattices>
</Phase>
"""
    body += f"""<Parameter Id="G({dis_phase},AG;0)" Expr="-1000;" Bibref="t" />
<Parameter Id="G({dis_phase},CU;0)" Expr="-2000;" Bibref="t" />
<Parameter Id="L({dis_phase},AG,CU;0)" Expr="-5000;" Bibref="t" />
"""
    one, two = ["AG:CU:CU:CU"], ["AG:AG:CU:CU"]
    if not perm:
        one = sorted({":".join(p) for p in itertools.permutations(["AG", "CU", "CU", "CU"])})
        two = sorted({":".join(p) for p in itertools.permutations(["AG", "AG", "CU", "CU"])})
    body += "".join(f'<Parameter Id="G(FCC_L12,{c};0)" Expr="-3000;" Bibref="t" />\n' for c in one)
    body += "".join(f'<Parameter Id="G(FCC_L12,{c};0)" Expr="-4000;" Bibref="t" />\n' for c in two)
    return read(xtdb(body, elements=("AG", "CU")))


def test_permutation_and_inline_disordered_part_match_explicit_form():
    rng = np.random.default_rng(1)
    y = np.empty((20, 8))
    y[:, 0::2] = rng.uniform(0.02, 0.98, (20, 4))
    y[:, 1::2] = 1 - y[:, 0::2]
    explicit = gm(_ordered(perm=False, synthesized=False), "FCC_L12", ["AG", "CU"], y)
    for perm, synth in ((True, False), (False, True), (True, True)):
        g = gm(_ordered(perm=perm, synthesized=synth), "FCC_L12", ["AG", "CU"], y)
        np.testing.assert_allclose(g, explicit, rtol=0, atol=1e-8)
    # the ordering terms do something: near 0 K (no entropy) L12 AgCu3 lies below disorder
    l12 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    dis = np.array([0.25, 0.75] * 4)
    db = _ordered(perm=True, synthesized=True)
    assert gm(db, "FCC_L12", ["AG", "CU"], l12, T=10.0) < gm(db, "FCC_L12", ["AG", "CU"], dis,
                                                              T=10.0) - 100


SIGMA_NO_SUBTRACT = """<Phase Id="SIGMA" Configuration="CEF" State="S" >
  <Sublattices NumberOf="2" Multiplicities="10 20" >
    <Constituents Sublattice="1" List="AG CU" />
    <Constituents Sublattice="2" List="AG CU" />
  </Sublattices>
  <AmendPhase><DisorderedPart Sum="2" /></AmendPhase>
</Phase>
<Parameter Id="G(SIGMA,AG;0)" Expr="-500;" Bibref="t" />
<Parameter Id="G(SIGMA,CU;0)" Expr="-700;" Bibref="t" />
<Parameter Id="L(SIGMA,AG,CU;0)" Expr="-9000;" Bibref="t" />
<Parameter Id="G(SIGMA,AG:CU;0)" Expr="-1000;" Bibref="t" />
<Parameter Id="G(SIGMA,CU:AG;0)" Expr="2000;" Bibref="t" />"""

SIGMA_TDB = """ELEMENT AG BLANK 1 0 0 !
ELEMENT CU BLANK 1 0 0 !
TYPE_DEFINITION P GES A_P_D SIGMA NEVER_DIS SIGMA_DIS !
PHASE SIGMA_DIS % 1 30 !
CONSTITUENT SIGMA_DIS :AG,CU: !
PHASE SIGMA %P 2 10 20 !
CONSTITUENT SIGMA :AG,CU:AG,CU: !
PARAMETER G(SIGMA_DIS,AG;0) 1 -500; 6000 N !
PARAMETER G(SIGMA_DIS,CU;0) 1 -700; 6000 N !
PARAMETER L(SIGMA_DIS,AG,CU;0) 1 -9000; 6000 N !
PARAMETER G(SIGMA,AG:CU;0) 1 -1000; 6000 N !
PARAMETER G(SIGMA,CU:AG;0) 1 2000; 6000 N !
"""


@pytest.mark.skipif(not _supports_never_disorder(), reason="needs pycalphad 0.11.2 or later")
def test_disordered_part_without_subtract_matches_tdb_never_disorder():
    db = read(xtdb(SIGMA_NO_SUBTRACT, elements=("AG", "CU")))
    assert db.phases["SIGMA"].model_hints.get("never_disorder") is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = Database(SIGMA_TDB)
    rng = np.random.default_rng(2)
    y = np.empty((15, 4))
    y[:, 0::2] = rng.uniform(0.02, 0.98, (15, 2))
    y[:, 1::2] = 1 - y[:, 0::2]
    np.testing.assert_allclose(gm(db, "SIGMA", ["AG", "CU"], y), gm(ref, "SIGMA", ["AG", "CU"], y),
                               rtol=0, atol=1e-8)
    assert 'Subtract' not in pycalphad_xtdb.xtdb_text(db).split('<Phase Id="SIGMA"')[1].split("</Phase>")[0]


@pytest.mark.skipif(_supports_never_disorder(), reason="pycalphad supports never_disorder")
def test_disordered_part_without_subtract_drops_the_phase_on_old_pycalphad():
    with pytest.warns(UserWarning, match="SIGMA.*0.11.2"):
        db = Database.from_string(xtdb(SIGMA_NO_SUBTRACT, elements=("AG", "CU")), fmt="xtdb")
    assert "SIGMA" not in db.phases


def test_unsupported_configuration_and_model_warn():
    body = LIQUID.replace('Configuration="CEF"', 'Configuration="MQMQA"')
    with pytest.warns(UserWarning, match="MQMQA"):
        Database.from_string(xtdb(body), fmt="xtdb")
    body = LIQUID.replace("</Sublattices>", '</Sublattices>\n  <AmendPhase Models="NEWMODEL" />')
    with pytest.warns(UserWarning, match="NEWMODEL"):
        Database.from_string(xtdb(body), fmt="xtdb")


def test_custom_model_parameter_identifiers():
    body = """<ModelDescriptions Software="test">
  <Magnetic Id="IHJBCC" MPID1="BMOM" MPID2="TCUR" Aff="-1" Bibref="t" />
</ModelDescriptions>
<Phase Id="BCC_A2" Configuration="CEF" State="S" >
  <Sublattices NumberOf="2" Multiplicities="1 3" >
    <Constituents Sublattice="1" List="CU" />
    <Constituents Sublattice="2" List="VA" />
  </Sublattices>
  <AmendPhase Models="IHJBCC" />
</Phase>
<Parameter Id="TCUR(BCC_A2,CU:VA;0)" Expr="1043;" Bibref="t" />
<Parameter Id="BMOM(BCC_A2,CU:VA;0)" Expr="2.22;" Bibref="t" />"""
    db = read(xtdb(body))
    hints = db.phases["BCC_A2"].model_hints
    assert hints["ihj_magnetic_afm_factor"] == -1 and hints["ihj_magnetic_structure_factor"] == 0.4
    kinds = {p["parameter_type"] for p in db._parameters.all()}
    assert kinds == {"TC", "BMAGN"}


def test_stated_gas_constant_is_used():
    body = LIQUID + """<TPfun Id="R" Expr="8.31451;" />
<Parameter Id="G(LIQUID,AG;0)" Expr="R*T;" Bibref="t" />"""
    db = read(xtdb(body))
    assert gm(db, "LIQUID", ["AG"], [1.0], T=1000.0) == pytest.approx(8314.51, abs=1e-8)


def test_species_stoichiometry_and_charge():
    body = """<Species Id="CU2O" Stoichiometry="CU2O1" />
<Species Id="CU+2" Stoichiometry="CU/+2" />
<Species Id="O-2" Stoichiometry="O/-2" />
<Species Id="AGCU1/3" Stoichiometry="AG1CU1/3" />"""
    db = read(xtdb(body, elements=("AG", "CU", "O")))
    sp = {s.name: s for s in db.species}
    assert sp["CU2O"].constituents == {"CU": 2.0, "O": 1.0} and sp["CU2O"].charge == 0
    assert sp["CU+2"].charge == 2 and sp["O-2"].charge == -2
    assert sp["AGCU1/3"].constituents["CU"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("text, message", [
    ("<TDB/>", "XTDB"),
    ("<XTDB><Phase></XTDB>", "well-formed"),
    (xtdb(LIQUID + '<Parameter Id="G(LIQUID,AG;0)" Expr="GEIN(THETA);" Bibref="t" />'),
     "numeric"),
    (xtdb(LIQUID + '<Parameter Id="G LIQUID AG" Expr="1;" Bibref="t" />'), "parameter Id"),
])
def test_bad_files_fail_loudly(text, message):
    with pytest.raises(ValueError, match=message):
        Database.from_string(text, fmt="xtdb")


def test_mobility_parameter_keeps_its_diffusing_species():
    body = LIQUID + '<Parameter Id="MQ(LIQUID&amp;CU,AG;0)" Expr="-150000;" Bibref="t" />'
    db = read(xtdb(body))
    (p,) = db._parameters.all()
    assert p["parameter_type"] == "MQ" and p["diffusing_species"].name == "CU"
    assert "MQ(LIQUID&amp;CU,AG;0)" in pycalphad_xtdb.xtdb_text(db)


def test_unreadable_unused_function_is_skipped_with_warning():
    body = LIQUID + '<TPfun Id="BROKEN" Expr="R*T*LN(1.0E-5)*P);" />'
    with pytest.warns(UserWarning, match="BROKEN"):
        db = Database.from_string(xtdb(body), fmt="xtdb")
    assert "BROKEN" not in db.symbols


def test_parameter_using_a_missing_function_fails():
    body = LIQUID + """<TPfun Id="BROKEN" Expr="1+);" />
<Parameter Id="G(LIQUID,AG;0)" Expr="BROKEN;" Bibref="t" />"""
    with pytest.raises(ValueError, match="BROKEN, which could not be read"):
        read(xtdb(body))
    body = LIQUID + '<Parameter Id="G(LIQUID,AG;0)" Expr="GHSERAG;" Bibref="t" />'
    with pytest.warns(UserWarning, match=r"GHSERAG is used by G\(LIQUID"):
        Database.from_string(xtdb(body), fmt="xtdb")


def test_global_models_pycalphad_does_not_apply_warn():
    body = LIQUID + '<TernaryXpol Phase="LIQUID" Constituents="AG AU CU" Xpol="KT3T3" Bibref="t" />'
    with pytest.warns(UserWarning, match="Muggianu"):
        Database.from_string(xtdb(body), fmt="xtdb")
    with pytest.warns(UserWarning, match="equal entropy"):
        Database.from_string(xtdb(LIQUID + '<EEC Id="EEC" Bibref="t" />'), fmt="xtdb")


def test_append_files_are_read(tmp_path):
    main = xtdb(LIQUID + '<AppendXTDB Parameters="extra.xtdb" />')
    (tmp_path / "main.xtdb").write_text(main)
    (tmp_path / "extra.xtdb").write_text(
        '<XTDB><TPfun Id="GAG" Expr="-1000;" />'
        '<Parameter Id="G(LIQUID,AG;0)" Expr="GAG;" Bibref="t" /></XTDB>')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database(str(tmp_path / "main.xtdb"))
    assert gm(db, "LIQUID", ["AG"], [1.0]) == pytest.approx(-1000.0, abs=1e-9)
