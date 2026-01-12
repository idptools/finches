"""
Script to generate test data files for test_epsilon_calculation.py

Run this script from the tests directory to create the .npz files in test_data/
for both Mpipi_GGv1 and CALVADOS2 forcefields.
"""

import os
import numpy as np

from finches import epsilon_stateless as epsilon_calculation
from finches.epsilon_calculation import InteractionMatrixConstructor
from finches.forcefields.mpipi import Mpipi_model
from finches.forcefields.calvados import calvados_model


# Test sequences - must match test_epsilon_calculation.py
TEST_SEQUENCES = [
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLP"
    "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQLTLPLPN",
    "GSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGS",
    "EEEEEEEEEEEKKKKKKKKKKKEEEEEEEEEEEKKKKKKKKKKKEEEEEEEEEE",
    "LLLLLAAAAAEKEKEKEAAAALELLLYYYYYSSSSSSSQSQSQPSQPLSLLQSQ"
]
NAMES = ["t0", "t1", "t2", "t3"]


def generate_data_for_model(X, prefix):
    """Generate all test data for a given model."""
    
    print(f"Generating data for {prefix}...")
    
    epsilon_vectors = {}
    epsilon_values = {}
    
    for i, t in enumerate(TEST_SEQUENCES):
        n = NAMES[i]
        
        # NOWEIGHTING vectors
        attr_vec, rep_vec = epsilon_calculation.get_sequence_epsilon_vectors(
            t, t, X, use_charge_weighting=False, use_aliphatic_weighting=False)
        epsilon_vectors[f"{n}_NOWEIGHTING"] = [attr_vec, rep_vec]
        
        # Vectors with custom charge prefactor
        attr_vec, rep_vec = epsilon_calculation.get_sequence_epsilon_vectors(
            t, t, X, charge_prefactor=0.25)
        epsilon_vectors[f"{n}_charge_prefactor_25"] = [attr_vec, rep_vec]

        # Vectors with custom null interaction baseline
        attr_vec, rep_vec = epsilon_calculation.get_sequence_epsilon_vectors(
            t, t, X, null_interaction_baseline=-0.15)
        epsilon_vectors[f"{n}_null_baseline_neg15"] = [attr_vec, rep_vec]

        # Epsilon values
        epsilon_values[f"{n}_DEFAULT"] = epsilon_calculation.get_sequence_epsilon_value(t, t, X)
        epsilon_values[f"{n}_NOCHARGE"] = epsilon_calculation.get_sequence_epsilon_value(
            t, t, X, use_charge_weighting=False)
        epsilon_values[f"{n}_NOALIPHATICS"] = epsilon_calculation.get_sequence_epsilon_value(
            t, t, X, use_aliphatic_weighting=False)
    
    np.savez(f"test_data/{prefix}_seq_epsilon_and_vectors.npz",
             epsilon_vectors=epsilon_vectors, epsilon_values=epsilon_values)
    print(f"  Saved: test_data/{prefix}_seq_epsilon_and_vectors.npz")
    
    # Generate sliding epsilon data
    sliding_data = {}
    for i, t in enumerate(TEST_SEQUENCES):
        n = NAMES[i]
        sliding_data[f"sliding_{n}_w1"], _, _ = X.calculate_sliding_epsilon(t, t, window_size=1)
        sliding_data[f"sliding_{n}_default"], _, _ = X.calculate_sliding_epsilon(t, t, window_size=31)
        sliding_data[f"sliding_{n}_w15"], _, _ = X.calculate_sliding_epsilon(t, t, window_size=15)
    
    np.savez(f"test_data/{prefix}_sliding_epsilon.npz", **sliding_data)
    print(f"  Saved: test_data/{prefix}_sliding_epsilon.npz")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    os.makedirs("test_data", exist_ok=True)
    
    # Generate data for Mpipi_GGv1
    mpipi_params = Mpipi_model(version="Mpipi_GGv1")
    X_mpipi = InteractionMatrixConstructor(parameters=mpipi_params)
    generate_data_for_model(X_mpipi, "Mpipi_GGv1")
    
    # Generate data for CALVADOS2
    calvados_params = calvados_model(version="CALVADOS2")
    X_calvados = InteractionMatrixConstructor(parameters=calvados_params)
    generate_data_for_model(X_calvados, "CALVADOS2")
    
    print("\nTest data generation complete!")


if __name__ == "__main__":
    main()
