import pandas as pd
import pytest

import finches
from finches import epsilon_calculation
from finches.forcefields.calvados import calvados_model
from finches.forcefields.mpipi import Mpipi_model

from ..test_data.test_sequences import t0, test_sequences

# import un


# test are done in the context with the mPiPi_GGv1 model
L_model = Mpipi_model("mPiPi_GGv1")
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
