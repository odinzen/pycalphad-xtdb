"""pycalphad Model with the third-generation conventions of the XTDB specification."""

from symengine import S, exp, log
from tinydb import where

from pycalphad import Model
from pycalphad import variables as v


# Set by the reader on phases whose THETA parameters came from XTDB LNTH (one oscillator per
# atom). Phases without it, such as those read from a TDB file, keep the TDB meaning.
PER_ATOM_HINT = "einstein_per_atom"


def multi_atom_theta_phases(dbe):
    """Phases with an Einstein THETA parameter on an endmember of more than one atom.

    This is where TDB and XTDB part ways: TDB's THETA adds one oscillator per formula unit (the
    Thermo-Calc convention pycalphad keeps on purpose, pycalphad#760), XTDB's LNTH one per atom.
    """
    multi = set()
    for param in dbe.search(where("parameter_type") == "THETA"):
        phase = dbe.phases[param["phase_name"]]
        atoms = sum(n for n, subl in zip(phase.sublattices, param["constituent_array"])
                    if not all(c.name in ("VA", "*") for c in subl))
        if abs(atoms - 1) > 1e-12:
            multi.add(phase.name)
    return sorted(multi)


def gas_constant(dbe):
    """The database's own R when it states one as a number, else pycalphad's."""
    value = dbe.symbols.get("R")
    try:
        return float(value)
    except (TypeError, RuntimeError, ValueError):
        return v.R


class XTDBModel(Model):
    """Model for third-generation (Einstein + two-state liquid) databases read from XTDB.

    XTDB's LNTH parameter counts one Einstein oscillator per atom (Sundman et al. 2025,
    Eq. 12), so per mole of atoms a compound such as Al4C3 (7 atoms) gets one full oscillator.
    pycalphad's Model keeps the TDB meaning of THETA, one oscillator per formula unit, which is
    the Thermo-Calc convention TDB files are written for (pycalphad#760). XTDBModel applies the
    per-atom rule only to phases the XTDB reader marked, so TDB-origin phases are unchanged.
    The Einstein and two-state terms use the gas constant the database states, so a file
    written with R = 8.31451 evaluates as its authors intended.
    """

    def einstein_energy(self, dbe):
        phase = dbe.phases[self.phase_name]
        query = (
            (where("phase_name") == phase.name)
            & (where("parameter_type") == "THETA")
            & (where("constituent_array").test(self._array_validity))
        )
        lntheta = self.redlich_kister_sum(phase, dbe.search, query)
        if lntheta == S.Zero:
            return S.Zero
        R = gas_constant(dbe)
        theta = exp(lntheta)
        energy = 1.5 * R * theta + 3 * R * v.T * log(1 - exp(-theta / v.T))
        if phase.model_hints.get(PER_ATOM_HINT):
            return energy
        return energy / self._site_ratio_normalization

    def twostate_energy(self, dbe):
        phase = dbe.phases[self.phase_name]
        query = (
            (where("phase_name") == phase.name)
            & (where("parameter_type") == "GD")
            & (where("constituent_array").test(self._array_validity))
        )
        gd = self.redlich_kister_sum(phase, dbe.search, query)
        if gd == S.Zero:
            return S.Zero
        R = gas_constant(dbe)
        return -R * v.T * log(1 + exp(-gd / (R * v.T))) / self._site_ratio_normalization
