"""XTDB (XML thermodynamic database) support for pycalphad.

Importing this package registers the "xtdb" format, so ``Database("file.xtdb")`` reads an
XTDB file and ``db.to_file("out.xtdb")`` writes one. Third-generation databases (Einstein
and two-state liquid) should be calculated with ``model=XTDBModel``.
"""

__version__ = "0.1.1"

from pycalphad import Database

from .model import XTDBModel
from .reader import read_xtdb
from .writer import write_xtdb, xtdb_text

Database.register_format("xtdb", read=read_xtdb, write=write_xtdb)

__all__ = ["XTDBModel", "read_xtdb", "write_xtdb", "xtdb_text", "__version__"]
