"""This module enables in-silico evolution of IDRs"""

# Add imports here

import os


from ._version import __version__

# front-end imports here
from finches.frontend.calvados_frontend import CALVADOS_frontend
from finches.frontend.mpipi_frontend import Mpipi_frontend


# code that allows access to the data directory
_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)
