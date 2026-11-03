"""pycalphad Model with the third-generation conventions of the XTDB specification."""

from symengine import S, exp, log
from tinydb import where

from pycalphad import Model
from pycalphad import variables as v


def gas_constant(dbe):
    """The database's own R when it states one as a number, else pycalphad's."""
    value = dbe.symbols.get("R")
    try:
        return float(value)
    except (TypeError, RuntimeError, ValueError):
        return v.R


class XTDBModel(Model):
    """Model for third-generation (Einstein + two-state liquid) databases.

    pycalphad's Model adds one Einstein oscillator per formula unit and then divides by
    the moles of atoms, so a compound such as Al4C3 (7 atoms) gets a seventh of its
    Einstein heat capacity. Sundman et al. (2025), Eq. (12), and He et al. (2021) weight
    the composition-dependent Einstein term by the moles of atoms per formula unit, which
    per mole of atoms leaves one full oscillator. Pure elements and one-site phases are
    unchanged. The Einstein and two-state terms also use the gas constant the database
    states, so a file written with R = 8.31451 evaluates as its authors intended.
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
        return 1.5 * R * theta + 3 * R * v.T * log(1 - exp(-theta / v.T))

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
