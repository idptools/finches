"""FINCHES: predicting chemical specificity in intrinsically disordered regions."""

import os

from finches.frontend.calvados_frontend import CALVADOS_frontend
from finches.frontend.mpipi_frontend import Mpipi_frontend

__all__ = ["CALVADOS_frontend", "Mpipi_frontend", "get_data", "__version__"]

# absolute path to the installed package; used for the version file and data
_ROOT = os.path.abspath(os.path.dirname(__file__))

# Generate _version.py if missing and in the Read the Docs environment
if os.getenv("READTHEDOCS") == "True" and not os.path.isfile(
    os.path.join(_ROOT, "_version.py")
):
    import versioningit

    __version__ = versioningit.get_version(os.path.join(_ROOT, ".."))
else:
    from ._version import __version__


def get_data(path):
    """
    Return the absolute path to a file in the FINCHES data directory.

    Parameters
    ----------
    path : str
        Path relative to finches/data (for example "calvados/calvados_residues_dict.pickle").

    Returns
    -------
    str
        The absolute path. The file is not checked for existence.
    """
    return os.path.join(_ROOT, "data", path)
