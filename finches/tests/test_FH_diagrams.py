import pytest
#import un

import pandas as pd

import finches 

from finches.forcefields.mpipi import mpipi_model
from finches.forcefields.calvados import calvados_model
from finches import epsilon_calculation

from ..test_data.test_sequences import test_sequences, t0

# test are done in the context with the mPiPi_GGv1 model
L_model = mPiPi_model('mPiPi_GGv1')
X_local = epsilon_calculation.Interaction_Matrix_Constructor(L_model)

############################################################################################
##                                                                                        ##
##                                                                                        ##
##                   TESTING Interaction_Matrix_Constructor Class Functions               ##
##                                                                                        ##
##                                                                                        ##
############################################################################################

# ..........................................................................................
#
#
def build_DIELECTRIC_dependent_phase_diagrams():
    pass 

