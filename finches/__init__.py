"""This module enables in-silico evolution of IDRs"""

# Add imports here

import os

from .evolve_sequences import *

from ._version import __version__


# code that allows access to the data directory
_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)
