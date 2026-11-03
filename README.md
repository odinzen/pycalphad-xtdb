# pycalphad-xtdb

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23928488.svg)](https://doi.org/10.5281/zenodo.23928488)

Read and write XTDB files in [pycalphad](https://pycalphad.org). XTDB is the XML database
format for CALPHAD proposed by Sundman et al., *Calphad* 90 (2025) 102849
([doi:10.1016/j.calphad.2025.102849](https://doi.org/10.1016/j.calphad.2025.102849)). It
carries the same parameters as a TDB file, plus the third-generation models (Einstein solids
and the two-state liquid) that TDB has no standard place for.

## Install

```bash
pip install git+https://github.com/odinzen/pycalphad-xtdb
```

Requires pycalphad 0.11 or later.

## Use

Importing the package registers the `xtdb` format with pycalphad:

```python
import pycalphad_xtdb
from pycalphad import Database, equilibrium, variables as v

db = Database("examples/AlC.xtdb")       # read
db.to_file("AlC-copy.xtdb")               # write
Database("some.tdb").to_file("some.xtdb") # convert a TDB file

eq = equilibrium(db, ["AL", "C", "VA"], list(db.phases),
                 {v.T: 2000, v.P: 101325, v.X("C"): 0.3, v.N: 1},
                 model=pycalphad_xtdb.XTDBModel)
```

## Third-generation databases need `XTDBModel`

The XTDB paper (Eq. 12) weights the composition-dependent Einstein term by the number of atoms
in the formula unit. pycalphad's `Model` adds one Einstein oscillator per formula unit, so every
endmember with more than one atom loses part of its vibrational energy. Pure elements are not
affected, so a melting point can look right while the compounds are wrong. For Al4C3 (7 atoms)
from the He et al. (2021) Al-C assessment:

| | `XTDBModel` | pycalphad `Model` |
|---|---|---|
| Cp at 300 K, J/(mol-atom K) | 2.00 R | -0.22 R |
| Cp at 1000 K, J/(mol-atom K) | 3.04 R | 0.51 R |
| Al4C3 decomposition | 2425 K (measured 2156 degC = 2429 K) | Al4C3 not stable at 2000 K |

`XTDBModel` is a subclass of `Model` with the Einstein term weighted as the paper specifies.
It also evaluates the Einstein and two-state terms with the gas constant the file states
(8.31451 in the published files) instead of pycalphad's 8.3145. The reader warns when a
database has Einstein parameters on multi-atom endmembers.

## What is translated

- Elements, species (with charge), phases with any number of sublattices, the CEF and I2SL
  (ionic liquid) configurations.
- `TPfun` and `Parameter` with `LowT`, `HighT` and `Trange` pieces, `Defaults`, and the `GEIN`
  function inside expressions.
- Model parameters G, L, TC, BMAGN, NT, LNTH (Einstein), GD (two-state liquid), mobility
  parameters in the TDB form `MQ(PHASE&SPECIES,...)`, and any other identifier pycalphad
  knows. A file's own `ModelDescriptions` can rename the identifiers.
- Magnetic models IHJBCC, IHJREST and IHJQX, the permutation models FCC4PERM and BCC4PERM,
  wildcards (`*`), and `DisorderedPart` with `Subtract="Y"`, whether the disordered parameters
  sit in a separate phase or in the ordered phase with fewer sublattices. In the second case
  the reader creates the disordered phase as `<PHASE>_DIS`; leave it out of phase lists.
- `AppendXTDB` files next to the main file.

Features pycalphad cannot evaluate are reported, never dropped silently:

| In the file | What happens |
|---|---|
| `Configuration="MQMQA"` or another non-CEF entropy model | phase left out, warning |
| `DisorderedPart` without `Subtract="Y"` (Eq. 15) | phase left out, warning |
| an unknown model in `AmendPhase` | warning, that contribution is missing |
| `TernaryXpol` (Kohler, Toop) | warning, pycalphad uses Muggianu |
| EEC (equal entropy criterion) | warning, not enforced |
| a function that cannot be parsed | warning if unused, error if a parameter uses it |
| a function that is used but never defined | warning, pycalphad evaluates it as zero |
| a parameter with the wrong number of sublattices | warning, parameter skipped |

## Choices where the paper leaves room

- **Ternary interaction degrees** follow the XTDB definition (paper Eq. 23): degree 0 is
  composition independent and degrees 1 to 3 are the v-weighted terms. TDB files use 0 to 2
  for the weighted terms. The reader and writer convert between the two, so a TDB file
  converted to XTDB and read back gives identical energies.
- **Reciprocal parameters** keep the TDB meaning used by Thermo-Calc, OpenCalphad and
  pycalphad, where degree 1 multiplies the site-fraction difference on the second sublattice.
  Eq. 24 of the paper writes the two the other way round; the OpenCalphad source follows the
  TDB meaning.
- **IHJQX** (the Xiong magnetic model) is read with structure factor 0.25, the value behind the
  coefficients in the reference ModelDescriptions file.
- **Binary Redlich-Kister terms** use alphabetical constituent order, as pycalphad does for
  TDB files.
- `GEIN(theta)` calls are stored as functions named `GEIN_<theta>` (for example
  `GEIN_1953P3`) and written back as `GEIN(1953.3)`.
- Bibliography text is not kept in a pycalphad `Database`; the writer lists each reference by
  its Id.

## Validation

- Every TDB file shipped with pycalphad 0.11.1 (35 files, including COST507) is written as
  XTDB, read back, and every phase pycalphad can build is compared at 60 random site-fraction
  points and three temperatures. Gibbs energies agree within 1e-6 J/mol.
- `examples/AlC.xtdb` gives the same Gibbs energy as the Al-C example of the XTDB authors
  (OpenCalphad) for all seven phases, and its endmember energies match an independent
  evaluator to 1e-6 J/mol. A test plants a 50 J/mol error to show the check can fail.
- Tests check the ternary and wildcard formulas, permutations and both forms of the disordered
  part against hand-written expressions, and the failure modes in the table above.

```bash
pip install -e ".[test]"
pytest
```

## Example databases

`examples/AlC.xtdb`: Al-C, He, Kaplan, Mao, Selleby, *Calphad* 72 (2021) 102250
([doi:10.1016/j.calphad.2021.102250](https://doi.org/10.1016/j.calphad.2021.102250)).
`examples/PbSn.xtdb`: Pb-Sn, Ngai and Chang, *Calphad* 5 (1981) 267-276
([doi:10.1016/0364-5916(81)90009-2](https://doi.org/10.1016/0364-5916(81)90009-2)).
Both are our own transcriptions of the published parameters and are released under CC BY 4.0.
Cite the original assessments when you use them.

## License and citation

Code: MIT, copyright Odinzen LLC. Example data: CC BY 4.0. If you use this package, please
cite it (Zenodo, [doi:10.5281/zenodo.23928488](https://doi.org/10.5281/zenodo.23928488), all
versions; 0.1.0 is [doi:10.5281/zenodo.23928489](https://doi.org/10.5281/zenodo.23928489)), and
cite the XTDB paper for the format.
