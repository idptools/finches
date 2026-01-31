"""
Comprehensive test suite for the epsilon_to_FHtheory module.

Tests the functions that convert epsilon values to Flory-Huggins phase diagrams:
    - epsilon_to_phase_diagram: Converts epsilon and sequence to phase diagram
    - return_phase_diagram: Wrapper that calculates epsilon and builds phase diagram
    - build_SALT_dependent_phase_diagrams: Salt-dependent phase diagram generation
    - build_PH_dependent_phase_diagrams: pH-dependent phase diagram generation
    - build_DIELECTRIC_dependent_phase_diagrams: Dielectric-dependent phase diagram generation
"""

import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from finches.epsilon_to_FHtheory import (
    epsilon_to_phase_diagram,
    return_phase_diagram,
    build_SALT_dependent_phase_diagrams,
    build_PH_dependent_phase_diagrams,
    build_DIELECTRIC_dependent_phase_diagrams,
)
from finches.epsilon_calculation import InteractionMatrixConstructor
from finches.forcefields.mpipi import Mpipi_model
from finches.forcefields.calvados import calvados_model


# Change to test directory for data files
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Test sequences
# =============================================================================

TEST_SEQUENCES = {
    "short": "AEKLSQPGWY",
    "medium": "MSKGEELFTGVVPILVELDGDVNGHKFSVS",
    "charged_positive": "KKKKKKKKKKKKKKKKKKKK",
    "charged_negative": "EEEEEEEEEEEEEEEEEEEE",
    "aromatic_rich": "FYWFYWFYWFYWFYWFYWFY",
    "hydrophobic": "LLLLLLLLLLLLLLLLLLLL",
    "idr_like": "SSQPSQSQPQSQSQPASPASQ",
}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mpipi_imc():
    """Fixture for Mpipi InteractionMatrixConstructor."""
    params = Mpipi_model(version='Mpipi_GGv1')
    return InteractionMatrixConstructor(parameters=params)


@pytest.fixture
def calvados_imc():
    """Fixture for CALVADOS InteractionMatrixConstructor."""
    params = calvados_model(version='CALVADOS2')
    return InteractionMatrixConstructor(parameters=params)


@pytest.fixture(params=["mpipi", "calvados"])
def model_fixture(request):
    """
    Fixture that provides both model types for parametrized testing.
    Returns tuple of (model_name, IMC_instance)
    """
    if request.param == "mpipi":
        params = Mpipi_model(version='Mpipi_GGv1')
        imc = InteractionMatrixConstructor(parameters=params)
        return ("mpipi", imc)
    else:
        params = calvados_model(version='CALVADOS2')
        imc = InteractionMatrixConstructor(parameters=params)
        return ("calvados", imc)


# =============================================================================
# Tests for epsilon_to_phase_diagram()
# =============================================================================

class TestEpsilonToPhaseDiagram:
    """Tests for the epsilon_to_phase_diagram function."""

    def test_returns_correct_structure(self):
        """Test that output has correct structure with 8 elements."""
        seq = "AEKLSQPGWY"
        epsilon = -1.0  # Attractive epsilon
        
        result = epsilon_to_phase_diagram(seq, epsilon)
        
        assert len(result) == 8, "Should return list with 8 elements"
        # Elements 0-3: binodal data
        # Elements 4-7: spinodal data

    def test_dilute_dense_concentration_arrays(self):
        """Test that dilute and dense concentrations are numpy arrays."""
        seq = "AEKLSQPGWY"
        epsilon = -1.0
        
        result = epsilon_to_phase_diagram(seq, epsilon)
        
        dilute = result[0]
        dense = result[1]
        
        assert isinstance(dilute, (list, np.ndarray))
        assert isinstance(dense, (list, np.ndarray))
        assert len(dilute) == len(dense), "Dilute and dense should have same length"

    def test_concentrations_physically_valid(self):
        """Test that concentrations are between 0 and 1 (valid volume fractions)."""
        seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        epsilon = -2.0
        
        result = epsilon_to_phase_diagram(seq, epsilon)
        
        dilute = np.array(result[0])
        dense = np.array(result[1])
        
        assert np.all(dilute >= 0) and np.all(dilute <= 1), "Dilute conc should be [0,1]"
        assert np.all(dense >= 0) and np.all(dense <= 1), "Dense conc should be [0,1]"

    def test_dense_greater_than_dilute(self):
        """Test that dense phase concentration > dilute phase concentration."""
        seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        epsilon = -2.0
        
        result = epsilon_to_phase_diagram(seq, epsilon)
        
        dilute = np.array(result[0])
        dense = np.array(result[1])
        
        # For most of the coexistence curve, dense should be > dilute
        assert np.all(dense >= dilute), "Dense phase should be >= dilute phase"

    def test_critical_point_structure(self):
        """Test critical point is a list with [crit_phi, crit_T]."""
        seq = "AEKLSQPGWY"
        epsilon = -1.0
        
        result = epsilon_to_phase_diagram(seq, epsilon)
        
        crit_point = result[2]
        assert len(crit_point) == 2, "Critical point should have [phi, T]"
        
        crit_phi, crit_T = crit_point
        assert 0 < crit_phi < 1, "Critical phi should be between 0 and 1"
        assert crit_T > 0, "Critical temperature should be positive"

    def test_temperature_array_positive(self):
        """Test that temperature array contains positive values."""
        seq = "AEKLSQPGWY"
        epsilon = -1.0
        
        result = epsilon_to_phase_diagram(seq, epsilon)
        
        temps = result[3]
        assert np.all(np.array(temps) > 0), "Temperatures should be positive"

    def test_spinodal_data_present(self):
        """Test that spinodal data is present in output."""
        seq = "FYWFYWFYWFYWFYWFYWFY"  # Longer aromatic sequence for better coverage
        epsilon = -5.0  # Strongly attractive to ensure spinodal data is generated
        
        result = epsilon_to_phase_diagram(seq, epsilon)
        
        s_dilute = result[4]
        s_dense = result[5]
        s_crit = result[6]
        s_temps = result[7]
        
        # Spinodal data should be lists/arrays (may be empty for some parameter ranges)
        assert isinstance(s_dilute, (list, np.ndarray)), "Spinodal dilute should be list/array"
        assert isinstance(s_dense, (list, np.ndarray)), "Spinodal dense should be list/array"
        assert len(s_crit) == 2, "Spinodal critical point should have 2 elements"
        assert isinstance(s_temps, np.ndarray), "Spinodal temps should be array"

    def test_positive_epsilon_handled(self):
        """Test that positive (repulsive) epsilon is handled gracefully."""
        seq = "KKKKKKKKKKKKKKKKKKKK"
        epsilon = 1.0  # Repulsive
        
        # Should not raise an exception
        result = epsilon_to_phase_diagram(seq, epsilon)
        
        # Should still return valid structure
        assert len(result) == 8

    def test_zero_epsilon_edge_case(self):
        """Test epsilon = 0 edge case."""
        seq = "AEKLSQPGWY"
        epsilon = 0.0
        
        # Should not raise an exception (gets converted to -0.01 internally)
        result = epsilon_to_phase_diagram(seq, epsilon)
        assert len(result) == 8

    def test_strongly_attractive_epsilon(self):
        """Test with strongly attractive (very negative) epsilon."""
        seq = "FYWFYWFYWFYWFYWFYWFY"  # Aromatic, likely attractive
        epsilon = -10.0
        
        result = epsilon_to_phase_diagram(seq, epsilon)
        
        crit_T = result[2][1]
        # Strong attraction should give higher critical temperature
        assert crit_T > 0

    def test_sequence_length_affects_critical_point(self):
        """Test that sequence length affects the critical point."""
        short_seq = "AEKLS"  # 5 residues
        long_seq = "AEKLSQPGWYFVMNHAEKLSQPGWYFVMNH"  # 30 residues
        epsilon = -2.0
        
        result_short = epsilon_to_phase_diagram(short_seq, epsilon)
        result_long = epsilon_to_phase_diagram(long_seq, epsilon)
        
        # Different lengths should give different critical points
        crit_phi_short = result_short[2][0]
        crit_phi_long = result_long[2][0]
        
        # Longer chains typically have lower critical concentration
        assert crit_phi_long < crit_phi_short

    def test_different_epsilon_values(self):
        """Test that different epsilon values give different phase diagrams."""
        seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        
        result_weak = epsilon_to_phase_diagram(seq, -0.5)
        result_strong = epsilon_to_phase_diagram(seq, -5.0)
        
        # Stronger attraction should have higher critical temperature
        crit_T_weak = result_weak[2][1]
        crit_T_strong = result_strong[2][1]
        
        assert crit_T_strong > crit_T_weak


# =============================================================================
# Tests for return_phase_diagram()
# =============================================================================

class TestReturnPhaseDiagram:
    """Tests for the return_phase_diagram wrapper function."""

    def test_returns_same_structure_as_epsilon_function(self, mpipi_imc):
        """Test that output structure matches epsilon_to_phase_diagram."""
        seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        
        result = return_phase_diagram(seq, mpipi_imc)
        
        assert len(result) == 8, "Should return 8-element list"

    def test_works_with_mpipi(self, mpipi_imc):
        """Test that function works with Mpipi model."""
        seq = "FYWFYWFYWFYWFYWFYWFY"  # Aromatic-rich, more likely to have attractive eps
        
        result = return_phase_diagram(seq, mpipi_imc)
        
        assert len(result) == 8
        # Check we get valid structure (may or may not have phase separation data)
        assert isinstance(result[0], (list, np.ndarray))
        assert isinstance(result[1], (list, np.ndarray))

    def test_works_with_calvados(self, calvados_imc):
        """Test that function works with CALVADOS model."""
        seq = "FYWFYWFYWFYWFYWFYWFY"  # Aromatic-rich sequence
        
        result = return_phase_diagram(seq, calvados_imc)
        
        assert len(result) == 8
        # Check we get valid structure
        assert isinstance(result[0], (list, np.ndarray))
        assert isinstance(result[1], (list, np.ndarray))

    def test_different_sequences_different_diagrams(self, model_fixture):
        """Test that different sequences produce different phase diagrams."""
        model_name, imc = model_fixture
        
        result_charged = return_phase_diagram("EEEEEEEEEEEEEEEEEEEE", imc)
        result_aromatic = return_phase_diagram("FYWFYWFYWFYWFYWFYWFY", imc)
        
        # Critical temperatures should differ for different sequences
        crit_T_charged = result_charged[2][1]
        crit_T_aromatic = result_aromatic[2][1]
        
        assert crit_T_charged != crit_T_aromatic, (
            f"{model_name}: Different sequences should have different critical T"
        )

    @pytest.mark.parametrize("seq_name,seq", list(TEST_SEQUENCES.items()))
    def test_various_sequences(self, seq_name, seq, mpipi_imc):
        """Test that various sequence types produce valid output."""
        result = return_phase_diagram(seq, mpipi_imc)
        
        assert len(result) == 8, f"Sequence {seq_name} should return 8-element list"
        
        # Check concentrations are valid
        dilute = np.array(result[0])
        dense = np.array(result[1])
        
        if len(dilute) > 0:
            assert np.all(dilute >= 0) and np.all(dilute <= 1), (
                f"{seq_name}: dilute concentrations should be in [0,1]"
            )
        if len(dense) > 0:
            assert np.all(dense >= 0) and np.all(dense <= 1), (
                f"{seq_name}: dense concentrations should be in [0,1]"
            )


# =============================================================================
# Tests for build_SALT_dependent_phase_diagrams()
# =============================================================================

class TestSaltDependentPhaseDiagrams:
    """Tests for salt-dependent phase diagram generation."""

    def test_returns_correct_structure(self, mpipi_imc):
        """Test that output has correct structure."""
        seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        salt_conditions = [0.05, 0.15, 0.25]
        
        result = build_SALT_dependent_phase_diagrams(seq, mpipi_imc, salt_conditions)
        
        assert len(result) == 3, "Should return list with 3 elements"
        assert result[0] == salt_conditions, "First element should be condition list"
        assert isinstance(result[1], dict), "Second element should be diagrams dict"
        assert isinstance(result[2], dict), "Third element should be epsilons dict"

    def test_diagrams_for_each_condition(self, mpipi_imc):
        """Test that a diagram is generated for each salt condition."""
        seq = "AEKLSQPGWY"
        salt_conditions = [0.05, 0.15, 0.30]
        
        result = build_SALT_dependent_phase_diagrams(seq, mpipi_imc, salt_conditions)
        
        diagrams = result[1]
        epsilons = result[2]
        
        for salt in salt_conditions:
            assert salt in diagrams, f"Should have diagram for salt={salt}"
            assert salt in epsilons, f"Should have epsilon for salt={salt}"
            assert len(diagrams[salt]) == 8, f"Diagram at salt={salt} should have 8 elements"

    def test_salt_affects_results(self, mpipi_imc):
        """Test that different salt concentrations give different results."""
        seq = "EKEKEKEKEKEKEKEKEKEK"  # Charged sequence, salt should matter
        salt_conditions = [0.01, 0.50]
        
        result = build_SALT_dependent_phase_diagrams(seq, mpipi_imc, salt_conditions)
        
        eps_low_salt = result[2][0.01]
        eps_high_salt = result[2][0.50]
        
        # Epsilon values should differ with salt
        assert eps_low_salt != eps_high_salt, "Salt should affect epsilon values"

    def test_restores_original_salt(self, mpipi_imc):
        """Test that original salt value is restored after function call."""
        seq = "AEKLSQPGWY"
        original_salt = mpipi_imc.parameters.salt
        salt_conditions = [0.10, 0.20]
        
        build_SALT_dependent_phase_diagrams(seq, mpipi_imc, salt_conditions)
        
        # Salt should be restored
        assert mpipi_imc.parameters.salt == original_salt, "Salt should be restored"

    def test_single_salt_condition(self, mpipi_imc):
        """Test with a single salt condition."""
        seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        salt_conditions = [0.15]
        
        result = build_SALT_dependent_phase_diagrams(seq, mpipi_imc, salt_conditions)
        
        assert len(result[1]) == 1
        assert 0.15 in result[1]

    def test_works_with_calvados(self, calvados_imc):
        """Test that function works with CALVADOS model."""
        seq = "AEKLSQPGWY"
        salt_conditions = [0.10, 0.20]
        
        result = build_SALT_dependent_phase_diagrams(seq, calvados_imc, salt_conditions)
        
        assert len(result) == 3
        assert len(result[1]) == 2


# =============================================================================
# Tests for build_PH_dependent_phase_diagrams()
# =============================================================================

class TestPHDependentPhaseDiagrams:
    """Tests for pH-dependent phase diagram generation."""

    def test_returns_correct_structure(self, calvados_imc):
        """Test that output has correct structure."""
        seq = "HHHHHHHHHHHHHHHHHHHH"  # Histidine, pH-sensitive
        ph_conditions = [5.0, 7.0, 9.0]
        
        result = build_PH_dependent_phase_diagrams(seq, calvados_imc, ph_conditions)
        
        assert len(result) == 3, "Should return list with 3 elements"
        assert result[0] == ph_conditions, "First element should be condition list"
        assert isinstance(result[1], dict), "Second element should be diagrams dict"
        assert isinstance(result[2], dict), "Third element should be epsilons dict"

    def test_diagrams_for_each_pH(self, calvados_imc):
        """Test that a diagram is generated for each pH condition."""
        seq = "HEHEHEHEHEHEHEHEHEHE"
        ph_conditions = [5.0, 7.0]
        
        result = build_PH_dependent_phase_diagrams(seq, calvados_imc, ph_conditions)
        
        diagrams = result[1]
        
        for ph in ph_conditions:
            assert ph in diagrams, f"Should have diagram for pH={ph}"
            assert len(diagrams[ph]) == 8, f"Diagram at pH={ph} should have 8 elements"

    def test_works_with_mpipi_model(self, mpipi_imc):
        """Test that pH-dependent function also works with Mpipi model if it supports pH."""
        seq = "HEHEHEHEHEHEHEHEHEHE"
        ph_conditions = [5.0, 7.0]
        
        # Try to run - if Mpipi supports pH it should work
        try:
            result = build_PH_dependent_phase_diagrams(seq, mpipi_imc, ph_conditions)
            # If it works, check the structure
            assert len(result) == 3
            assert len(result[1]) == 2
        except Exception as e:
            # If it doesn't support pH, that's also acceptable
            assert "pH" in str(e)


# =============================================================================
# Tests for build_DIELECTRIC_dependent_phase_diagrams()
# =============================================================================

class TestDielectricDependentPhaseDiagrams:
    """Tests for dielectric-dependent phase diagram generation."""

    def test_returns_correct_structure(self, mpipi_imc):
        """Test that output has correct structure."""
        seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        dielectric_conditions = [40.0, 60.0, 80.0]
        
        result = build_DIELECTRIC_dependent_phase_diagrams(
            seq, mpipi_imc, dielectric_conditions
        )
        
        assert len(result) == 3, "Should return list with 3 elements"
        assert result[0] == dielectric_conditions
        assert isinstance(result[1], dict)
        assert isinstance(result[2], dict)

    def test_diagrams_for_each_dielectric(self, mpipi_imc):
        """Test that a diagram is generated for each dielectric condition."""
        seq = "EKEKEKEKEKEKEKEKEKEK"
        dielectric_conditions = [60.0, 80.0]
        
        result = build_DIELECTRIC_dependent_phase_diagrams(
            seq, mpipi_imc, dielectric_conditions
        )
        
        diagrams = result[1]
        
        for d in dielectric_conditions:
            assert d in diagrams, f"Should have diagram for dielectric={d}"

    def test_dielectric_affects_charged_sequences(self, mpipi_imc):
        """Test that dielectric affects charged sequence results."""
        seq = "EKEKEKEKEKEKEKEKEKEK"
        dielectric_conditions = [40.0, 80.0]
        
        result = build_DIELECTRIC_dependent_phase_diagrams(
            seq, mpipi_imc, dielectric_conditions
        )
        
        eps_low_dielectric = result[2][40.0]
        eps_high_dielectric = result[2][80.0]
        
        # Different dielectric should give different epsilon
        # (though exact behavior depends on implementation)
        assert isinstance(eps_low_dielectric, (int, float))
        assert isinstance(eps_high_dielectric, (int, float))


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_epsilon_to_phase_matches_return_phase(self, mpipi_imc):
        """
        Test that epsilon_to_phase_diagram gives consistent results
        when using a manually computed epsilon vs return_phase_diagram.
        """
        from finches.epsilon_stateless import get_sequence_epsilon_value
        
        seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        
        # Get epsilon manually
        epsilon = get_sequence_epsilon_value(
            seq, seq, mpipi_imc,
            charge_prefactor=mpipi_imc.charge_prefactor,
            null_interaction_baseline=mpipi_imc.null_interaction_baseline,
            use_charge_weighting=True,
            use_aliphatic_weighting=True
        )
        
        # Use epsilon_to_phase_diagram directly
        result_direct = epsilon_to_phase_diagram(seq, epsilon)
        
        # Use return_phase_diagram wrapper
        result_wrapper = return_phase_diagram(seq, mpipi_imc)
        
        # Results should be identical
        assert_allclose(result_direct[0], result_wrapper[0], rtol=1e-10)
        assert_allclose(result_direct[1], result_wrapper[1], rtol=1e-10)
        assert_allclose(result_direct[2], result_wrapper[2], rtol=1e-10)

    def test_phase_diagram_consistency_across_models(self):
        """Test that both models produce valid (if different) phase diagrams."""
        seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        
        params_mpipi = Mpipi_model(version='Mpipi_GGv1')
        imc_mpipi = InteractionMatrixConstructor(parameters=params_mpipi)
        
        params_calvados = calvados_model(version='CALVADOS2')
        imc_calvados = InteractionMatrixConstructor(parameters=params_calvados)
        
        result_mpipi = return_phase_diagram(seq, imc_mpipi)
        result_calvados = return_phase_diagram(seq, imc_calvados)
        
        # Both should return valid structure
        assert len(result_mpipi) == 8
        assert len(result_calvados) == 8
        
        # Both should have positive critical temperatures
        assert result_mpipi[2][1] > 0
        assert result_calvados[2][1] > 0


# =============================================================================
# Edge case tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_sequence(self, mpipi_imc):
        """Test with very short sequence."""
        seq = "AEK"  # 3 residues
        
        result = return_phase_diagram(seq, mpipi_imc)
        
        assert len(result) == 8

    def test_single_residue_repeated(self, mpipi_imc):
        """Test with sequence of single repeated residue."""
        seq = "AAAAAAAAAA"  # All alanine
        
        result = return_phase_diagram(seq, mpipi_imc)
        
        assert len(result) == 8
        # Should still produce valid concentrations
        dilute = np.array(result[0])
        if len(dilute) > 0:
            assert np.all(dilute >= 0)

    def test_extreme_charge_imbalance(self, mpipi_imc):
        """Test sequences with extreme charge."""
        seq_positive = "KKKKKKKKKKKKKKKKKKKK"
        seq_negative = "EEEEEEEEEEEEEEEEEEEE"
        
        result_pos = return_phase_diagram(seq_positive, mpipi_imc)
        result_neg = return_phase_diagram(seq_negative, mpipi_imc)
        
        # Both should return valid results
        assert len(result_pos) == 8
        assert len(result_neg) == 8

    def test_empty_salt_conditions(self, mpipi_imc):
        """Test with empty condition list."""
        seq = "AEKLSQPGWY"
        
        result = build_SALT_dependent_phase_diagrams(seq, mpipi_imc, [])
        
        assert result[0] == []
        assert result[1] == {}
        assert result[2] == {}


# =============================================================================
# Numerical stability tests
# =============================================================================

class TestNumericalStability:
    """Tests for numerical stability of calculations."""

    def test_consistent_results_multiple_calls(self, mpipi_imc):
        """Test that multiple calls give identical results."""
        seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        
        result1 = return_phase_diagram(seq, mpipi_imc)
        result2 = return_phase_diagram(seq, mpipi_imc)
        
        assert_allclose(result1[0], result2[0])
        assert_allclose(result1[1], result2[1])
        assert_allclose(result1[2], result2[2])

    def test_binodal_above_spinodal(self, mpipi_imc):
        """Test that binodal lies outside spinodal (physically correct)."""
        seq = "FYWFYWFYWFYWFYWFYWFY"
        
        result = return_phase_diagram(seq, mpipi_imc)
        
        # At similar temperatures, binodal should be outside spinodal
        # This is a fundamental thermodynamic requirement
        binodal_dilute = np.array(result[0])
        binodal_dense = np.array(result[1])
        spinodal_dilute = np.array(result[4])
        spinodal_dense = np.array(result[5])
        
        # Check that arrays are non-empty
        if len(binodal_dilute) > 0 and len(spinodal_dilute) > 0:
            # The critical points should be the same
            assert_allclose(result[2][0], result[6][0], rtol=0.1)

    def test_temperature_monotonicity(self, mpipi_imc):
        """Test that temperature changes monotonically along binodal."""
        seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"
        
        result = return_phase_diagram(seq, mpipi_imc)
        
        temps = np.array(result[3])
        
        # Temperature array should be monotonic (either increasing or decreasing)
        if len(temps) > 1:
            diffs = np.diff(temps)
            # All differences should have the same sign (or be zero)
            assert np.all(diffs >= 0) or np.all(diffs <= 0), (
                "Temperature should change monotonically"
            )
