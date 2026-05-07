import pytest
#import un

import pandas as pd

import finches 

from finches.forcefields.mpipi import Mpipi_model
from finches.forcefields.calvados import calvados_model
from finches import epsilon_calculation

from finches.tests.test_data.test_sequences import test_sequences, t0

# test are done in the context with the Mpipi_GGv1 model
L_model = Mpipi_model('Mpipi_GGv1')
X_local = epsilon_calculation.InteractionMatrixConstructor(L_model)

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

