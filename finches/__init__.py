"""This module enables in-silico evolution of IDRs"""

# Add imports here

import os


# Generate _version.py if missing and in the Read the Docs environment
if os.getenv("READTHEDOCS") == "True" and not os.path.isfile('../finches/_version.py'):   
    import versioningit            
__version__ = versioningit.get_version('../')
else:
    from ._version import __version__

# front-end imports here
from finches.frontend.calvados_frontend import CALVADOS_frontend
from finches.frontend.mpipi_frontend import Mpipi_frontend


# code that allows access to the data directory
_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)
