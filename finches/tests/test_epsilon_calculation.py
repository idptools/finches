import pandas as pd
import pytest

import finches
from finches import epsilon_calculation
from finches.forcefields.calvados import calvados_model
from finches.forcefields.mpipi import Mpipi_model

from .test_data.test_sequences import t0, test_sequences

# import un


# test are done in the context with the mPiPi_GGv1 model
L_model = mpipi_model("mPiPi_GGv1")
X_local = epsilon_calculation.Interaction_Matrix_Constructor(L_model)

import numpy as np

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
def test_Interaction_Matrix_Constructor():
    pass


# ..........................................................................................
#
#
def test_calculate_pairwise_homotypic_matrix():
    TRUE_matrixes = np.load(
        "test_data/test_mPiPi_GGv1_homotypic_matrix.npz", allow_pickle=True
    )

    for i, t in enumerate(test_sequences):
        test_array = X_local.calculate_pairwise_homotypic_matrix(t)

        # expect this file to match precomputed homotypic matrix
        assert np.allclose(test_array, TRUE_matrixes["arr_0"][i])


# ..........................................................................................
#
#
def test_calculate_pairwise_heterotypic_matrix():
    TRUE_matrixes = np.load(
        "test_data/test_mPiPi_GGv1_heterotypic_matrix.npz", allow_pickle=True
    )

    for i, t in test_sequences:
        test_array = X_local.calculate_pairwise_heterotypic_matrix(t, t0)

        # expect this file to match precomputed heterotypic matrix
        assert np.allclose(test_array, TRUE_matrixes["arr_0"][i])


# ..........................................................................................
#
#
def test_calculate_weighted_pairwise_matrix():
    TRUE_matrixes = np.load(
        "test_data/test_mPiPi_GGv1_weighted_matrix.npz", allow_pickle=True
    )

    for i, t in test_sequences:
        # test defaults
        test_array = X_local.calculate_weighted_pairwise_matrix(t, t0)
        assert np.allclose(test_array, TRUE_matrixes["DEFAULT"][i])

        # test NO CHARGE weighting
        test_array = X_local.calculate_weighted_pairwise_matrix(
            t, t0, use_charge_weighting=False
        )
        assert np.allclose(test_array, TRUE_matrixes["NOCHARGE"][i])

        # test NO ALIPHATICS weighting
        test_array = X_local.calculate_weighted_pairwise_matrix(
            t, t0, use_aliphatic_weighting=False
        )
        assert np.allclose(test_array, TRUE_matrixes["NOALIPHATICS"][i])


############################################################################################
##                                                                                        ##
##                                                                                        ##
##                      TESTING general matrix manipulation Functions                     ##
##                                                                                        ##
##                                                                                        ##
############################################################################################


# ..........................................................................................
#
#
def test_get_attractive_repulsive_matrixes():
    TRUE_matrixes = np.load("test_data/test_matrix_manipulation.npz", allow_pickle=True)

    # compare the heterotypic DEFAULT matrix of test1:t0
    test_matrix = TRUE_matrixes["test_matrix"]
    TRUEattractive_matrix = TRUE_matrixes["attractive_repulsive_matrixes"][0]
    TRUErepulsive_matrix = TRUE_matrixes["attractive_repulsive_matrixes"][1]

    attractive_matrix, repulsive_matrix = (
        epsilon_calculation.get_attractive_repulsive_matrixes(test_matrix, -0.15)
    )

    assert np.allclose(attractive_matrix, TRUEattractive_matrix)
    assert np.allclose(repulsive_matrix, TRUErepulsive_matrix)


# ..........................................................................................
#
#
def test_mask_matrix():
    TRUE_matrixes = np.load("test_data/test_matrix_manipulation.npz", allow_pickle=True)

    # compare the heterotypic DEFAULT matrix of test1:t0
    test_matrix = TRUE_matrixes["test_matrix"]
    mask = TRUE_matrixes["bionary_mask"]
    TRUE_masked_matrix = TRUE_matrixes["masked_matrix"]

    # check masking
    test_masked_matrix = epsilon_calculation.masked_matrix(test_matrix, mask)
    assert np.allclose(test_masked_matrix, TRUE_masked_matrix)

    # check inpropper mask size
    with pytest.raises(
        Exception, match="column_mask and matrix are not the same shape"
    ):
        epsilon_calculation.masked_matrix(test_matrix, np.array([0, 0, 0, 0]))


# ..........................................................................................
#
#
def test_flatten_matrix_to_vector():
    TRUE_matrixes = np.load("test_data/test_matrix_manipulation.npz", allow_pickle=True)

    # compare vectors to truth
    test_matrix = TRUE_matrixes["test_matrix"]
    TRUE_vector0 = TRUE_matrixes["flattened_vector"][0]
    TRUE_vector1 = TRUE_matrixes["flattened_vector"][1]

    vector_o0 = epsilon_calculation.flatten_matrix_to_vector(test_matrix, orientation=0)
    vector_o1 = epsilon_calculation.flatten_matrix_to_vector(test_matrix, orientation=1)

    assert np.allclose(vector_o0, TRUE_vector0)
    assert np.allclose(vector_o1, TRUE_vector1)


############################################################################################
##                                                                                        ##
##                                                                                        ##
##                      TESTING general epsilon_calculation Functions                     ##
##                                                                                        ##
##                                                                                        ##
############################################################################################


# ..........................................................................................
#
#
def test_get_sequence_epsilon_vectors():
    TRUE_matrixes = np.load(
        "test_data/mPiPi_GGv1_seq_epsilon_and_vectors.npz", allow_pickle=True
    )

    # test t0 with test1 and t0
    names = ["t", "t0"]
    for i, t in enumerate([test_sequences[1], t0]):
        n = names[i]

        # compare vectors to truth default
        [attractive_vector, repulsive_vector] = all_manipulated[f"{n}_t0_NOWEIGHTING"]
        TESTattractive_vector, TESTrepulsive_vector = (
            epsilon_calculation.get_sequence_epsilon_vectors(t, t0, X_local)
        )
        assert np.allclose(TESTattractive_vector, attractive_vector)
        assert np.allclose(TESTrepulsive_vector, repulsive_vector)

        # compare vectors instance with passed baseline
        [attractive_vector, repulsive_vector] = all_manipulated[
            f"{n}_t0_prefactor_baseline"
        ]
        TESTattractive_vector, TESTrepulsive_vector = (
            epsilon_calculation.get_sequence_epsilon_vectors(
                t, t0, X_local, prefactor=0.25, null_interaction_baseline=-0.15
            )
        )
        assert np.allclose(TESTattractive_vector, attractive_vector)
        assert np.allclose(TESTrepulsive_vector, repulsive_vector)

        # compare vectors instance with no weighting
        [attractive_vector, repulsive_vector] = all_manipulated[f"{n}_t0_NOWEIGHTING"]
        TESTattractive_vector, TESTrepulsive_vector = (
            epsilon_calculation.get_sequence_epsilon_vectors(
                t,
                t0,
                X_local,
                use_charge_weighting=False,
                use_aliphatic_weighting=False,
            )
        )
        assert np.allclose(TESTattractive_vector, attractive_vector)
        assert np.allclose(TESTrepulsive_vector, repulsive_vector)

        # check to make sure weighting in working and different from NOWEIGHTING
        pass
