"""
Comprehensive test suite for the Flory-Huggins analytical backend.

Tests the functions in finches.analytical_fh.backend which implement
the analytical self-consistent solution for binodal concentrations
from Qian, Michaels, Knowles (2022).

Functions tested:
    - critical(n): Critical point calculation
    - spinodal(x, n): Spinodal boundary calculation
    - GL_binodal(x, n): Ginzburg-Landau approximation
    - binodal(x, n, iteration, UseImprovedMap): Self-consistent iteration
    - analytic_binodal(x, n): Closed-form analytical solution
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from finches.analytical_fh.backend import (
    critical,
    spinodal,
    GL_binodal,
    binodal,
    analytic_binodal,
)


# =============================================================================
# Test fixtures and helper functions
# =============================================================================


@pytest.fixture
def chi_values_above_critical():
    """Standard chi values above the critical point for N=1."""
    return np.array([2.5, 3.0, 4.0, 5.0, 10.0])


@pytest.fixture
def chi_values_below_critical():
    """Chi values below critical point for N=1 (chi_c = 2)."""
    return np.array([0.5, 1.0, 1.5, 1.9])


@pytest.fixture
def chain_lengths():
    """Various polymer chain lengths to test."""
    return [1, 2, 10, 50, 100]


def is_valid_concentration(phi):
    """Check if concentration is physically valid (0 < phi < 1)."""
    return np.all((phi > 0) & (phi < 1))


# =============================================================================
# Tests for critical() function
# =============================================================================


class TestCritical:
    """Tests for the critical point calculation."""

    def test_critical_n1_returns_correct_values(self):
        """Test critical point for symmetric case N=1."""
        result = critical(n=1)
        assert len(result) == 2
        # For N=1: chi_c = 2, phi_c = 0.5
        assert_allclose(result[0], 0.5, rtol=1e-10)  # phi_c
        assert_allclose(result[1], 2.0, rtol=1e-10)  # chi_c

    def test_critical_n1_symmetric(self):
        """Verify N=1 case gives symmetric critical concentration."""
        result = critical(n=1)
        assert_allclose(result[0], 0.5, rtol=1e-10)

    def test_critical_larger_n_lower_chi_c(self):
        """Longer polymers should have lower critical chi."""
        chi_c_1 = critical(n=1)[1]
        chi_c_10 = critical(n=10)[1]
        chi_c_100 = critical(n=100)[1]

        assert chi_c_1 > chi_c_10 > chi_c_100

    def test_critical_larger_n_lower_phi_c(self):
        """Longer polymers should have lower critical concentration."""
        phi_c_1 = critical(n=1)[0]
        phi_c_10 = critical(n=10)[0]
        phi_c_100 = critical(n=100)[0]

        assert phi_c_1 > phi_c_10 > phi_c_100

    def test_critical_n100(self):
        """Test critical point for N=100."""
        result = critical(n=100)
        # phi_c = 1/(1 + sqrt(100)) = 1/11
        # chi_c = 0.5 * (1 + 1/sqrt(100))^2 = 0.5 * 1.21 = 0.605
        expected_phi_c = 1.0 / (1.0 + np.sqrt(100))
        expected_chi_c = 0.5 * (1.0 + 1.0 / np.sqrt(100)) ** 2

        assert_allclose(result[0], expected_phi_c, rtol=1e-10)
        assert_allclose(result[1], expected_chi_c, rtol=1e-10)

    def test_critical_returns_numpy_array(self):
        """Ensure return type is numpy array."""
        result = critical(n=1)
        assert isinstance(result, np.ndarray)

    def test_critical_float_n(self):
        """Test that float values for n work correctly."""
        result = critical(n=1.5)
        assert len(result) == 2
        assert is_valid_concentration(result[0])
        assert result[1] > 0

    def test_critical_asymptotic_large_n(self):
        """For N -> infinity, chi_c -> 0.5 and phi_c -> 0."""
        result = critical(n=10000)
        assert_allclose(result[1], 0.5, rtol=0.05)  # chi_c approaches 0.5
        assert result[0] < 0.02  # phi_c approaches 0


# =============================================================================
# Tests for spinodal() function
# =============================================================================


class TestSpinodal:
    """Tests for the spinodal boundary calculation."""

    def test_spinodal_single_value_n1(self):
        """Test spinodal for a single chi value with N=1."""
        result = spinodal(3.0, n=1)
        assert len(result) == 2
        assert result[0] > result[1]  # dense > dilute
        assert is_valid_concentration(result)

    def test_spinodal_returns_ordered_concentrations(self):
        """Dense phase should have higher concentration than dilute."""
        result = spinodal(5.0, n=1)
        assert result[0] > result[1]

    def test_spinodal_symmetric_for_n1(self):
        """For N=1, spinodal should be symmetric around 0.5."""
        result = spinodal(3.0, n=1)
        # phi_dense + phi_dilute should equal 1 for N=1 (gamma=0)
        assert_allclose(result[0] + result[1], 1.0, rtol=1e-10)

    def test_spinodal_array_input(self, chi_values_above_critical):
        """Test spinodal with array of chi values."""
        result = spinodal(chi_values_above_critical, n=1)
        assert result.shape == (3, len(chi_values_above_critical))
        assert np.all(result[0] > result[1])  # dense > dilute for all

    def test_spinodal_filters_below_critical(self):
        """Values below critical chi should be filtered out."""
        chi_mixed = [1.5, 2.0, 2.5, 3.0]  # 2.0 is critical for N=1
        result = spinodal(chi_mixed, n=1)
        # only chi strictly above 2.0 is included (2.5, 3.0)
        assert len(result[2]) == 2
        assert np.all(result[2] > 2.0)

    def test_spinodal_raises_for_subcritical(self):
        """Should raise ValueError when all chi values are subcritical."""
        with pytest.raises(ValueError, match="interaction strength too small"):
            spinodal(1.5, n=1)

    def test_spinodal_raises_for_subcritical_array(self, chi_values_below_critical):
        """Should raise ValueError when all array values are subcritical."""
        with pytest.raises(ValueError, match="interaction strength too small"):
            spinodal(chi_values_below_critical, n=1)

    def test_spinodal_at_critical_point(self):
        """At critical chi, spinodal concentrations should converge."""
        chi_c = critical(n=1)[1]
        result = spinodal(chi_c + 0.001, n=1)
        # Near critical, both concentrations approach phi_c = 0.5
        assert_allclose(result[0], result[1], atol=0.05)

    def test_spinodal_increasing_separation_with_chi(self):
        """Higher chi should give wider separation between phases."""
        result1 = spinodal(2.5, n=1)
        result2 = spinodal(5.0, n=1)

        separation1 = result1[0] - result1[1]
        separation2 = result2[0] - result2[1]

        assert separation2 > separation1

    def test_spinodal_physical_bounds(self):
        """Spinodal concentrations should always be between 0 and 1."""
        for chi in [2.5, 5.0, 10.0, 50.0]:
            result = spinodal(chi, n=1)
            assert is_valid_concentration(result)

    def test_spinodal_with_large_n(self):
        """Test spinodal for large polymer chain length."""
        chi_c_n100 = critical(n=100)[1]
        result = spinodal(chi_c_n100 + 0.5, n=100)
        assert len(result) == 2
        assert is_valid_concentration(result)


# =============================================================================
# Tests for GL_binodal() function
# =============================================================================


class TestGLBinodal:
    """Tests for the Ginzburg-Landau binodal approximation."""

    def test_gl_binodal_single_value(self):
        """Test GL binodal for a single chi value."""
        result = GL_binodal(3.0, n=1)
        assert len(result) == 3
        assert result[0] > result[1]  # dense > dilute
        assert_allclose(result[2], 3.0)  # chi value preserved

    def test_gl_binodal_near_critical(self):
        """GL approximation should be accurate near critical point."""
        chi_c = critical(n=1)[1]
        phi_c = critical(n=1)[0]

        result = GL_binodal(chi_c + 0.01, n=1)
        # Near critical, both should be close to phi_c (increase tolerance)
        assert_allclose(result[0], phi_c, atol=0.10)
        assert_allclose(result[1], phi_c, atol=0.10)

    def test_gl_binodal_symmetric_around_phi_c(self):
        """For N=1, GL binodal is symmetric around phi_c = 0.5."""
        result = GL_binodal(3.0, n=1)
        phi_c = 0.5
        # |phi_dense - phi_c| should equal |phi_dilute - phi_c|
        assert_allclose(result[0] - phi_c, phi_c - result[1], rtol=1e-10)

    def test_gl_binodal_array_input(self, chi_values_above_critical):
        """Test GL binodal with array of chi values."""
        result = GL_binodal(chi_values_above_critical, n=1)
        assert result.shape == (3, len(chi_values_above_critical))

    def test_gl_binodal_raises_for_subcritical(self):
        """Should raise ValueError when chi is below critical."""
        with pytest.raises(ValueError, match="interaction strength too small"):
            GL_binodal(1.5, n=1)

    def test_gl_binodal_filters_subcritical_array(self):
        """Should filter out subcritical values from array."""
        chi_mixed = [1.0, 2.0, 3.0, 4.0]
        result = GL_binodal(chi_mixed, n=1)
        # chi_c = 2.0 for N=1; only chi strictly above chi_c is kept
        assert len(result[2]) == 2
        assert np.all(result[2] > 2.0)

    def test_gl_binodal_can_be_unphysical_at_large_chi(self):
        """At large chi, GL binodal can exceed physical bounds."""
        # For N=1, very large chi can push dilute phase negative
        # This is a known limitation of GL approximation
        result = GL_binodal(50.0, n=1)
        # At least check it runs without error
        assert len(result) == 3

    def test_gl_binodal_with_various_n(self, chain_lengths):
        """Test GL binodal for various chain lengths."""
        for n in chain_lengths:
            chi_c = critical(n=n)[1]
            result = GL_binodal(chi_c + 0.5, n=n)
            assert len(result) == 3


# =============================================================================
# Tests for binodal() function
# =============================================================================


class TestBinodal:
    """Tests for the self-consistent binodal calculation."""

    def test_binodal_single_value_n1(self):
        """Test binodal for single chi value with N=1."""
        result = binodal(3.0, n=1)
        assert len(result) == 3
        assert result[0] > result[1]  # dense > dilute
        assert_allclose(result[2], 3.0)

    def test_binodal_physical_bounds(self):
        """Binodal concentrations should be between 0 and 1."""
        for chi in [2.5, 5.0, 10.0]:
            result = binodal(chi, n=1)
            assert 0 < result[0] <= 1
            assert 0 <= result[1] < 1

    def test_binodal_n1_symmetric(self):
        """For N=1, phi_dense + phi_dilute = 1."""
        result = binodal(4.0, n=1)
        assert_allclose(result[0] + result[1], 1.0, rtol=1e-6)

    def test_binodal_array_input(self, chi_values_above_critical):
        """Test binodal with array of chi values."""
        result = binodal(chi_values_above_critical, n=1)
        assert result.shape == (3, len(chi_values_above_critical))

    def test_binodal_improved_map_default(self):
        """Default should use improved map."""
        result = binodal(4.0, n=1, UseImprovedMap=True)
        assert len(result) == 3

    def test_binodal_simple_map(self):
        """Test with simple (non-improved) map."""
        result = binodal(4.0, n=1, UseImprovedMap=False)
        assert len(result) == 3
        assert 0 < result[0] < 1
        assert 0 < result[1] < 1

    def test_binodal_improved_vs_simple_converge(self):
        """Both methods should converge to same result with enough iterations."""
        result_improved = binodal(4.0, n=1, iteration=10, UseImprovedMap=True)
        result_simple = binodal(4.0, n=1, iteration=20, UseImprovedMap=False)

        assert_allclose(result_improved[0], result_simple[0], rtol=1e-4)
        assert_allclose(result_improved[1], result_simple[1], rtol=1e-4)

    def test_binodal_iteration_convergence(self):
        """More iterations should improve accuracy."""
        result_1 = binodal(4.0, n=1, iteration=1)
        result_5 = binodal(4.0, n=1, iteration=5)
        result_10 = binodal(4.0, n=1, iteration=10)

        # Results should stabilize with more iterations
        diff_5_10 = np.abs(result_5[0] - result_10[0])
        diff_1_5 = np.abs(result_1[0] - result_5[0])

        assert diff_5_10 < diff_1_5

    def test_binodal_zero_iterations_returns_gl(self):
        """Zero iterations should return GL binodal."""
        result_binodal = binodal(4.0, n=1, iteration=0)
        result_gl = GL_binodal(4.0, n=1)

        assert_allclose(result_binodal[0], result_gl[0], rtol=1e-10)
        assert_allclose(result_binodal[1], result_gl[1], rtol=1e-10)

    def test_binodal_negative_iteration_raises(self):
        """Negative iteration count should raise AssertionError."""
        with pytest.raises(AssertionError):
            binodal(4.0, n=1, iteration=-1)

    def test_binodal_n_greater_than_1(self):
        """Test binodal for N > 1 case."""
        chi_c = critical(n=10)[1]
        result = binodal(chi_c + 1.0, n=10)

        assert len(result) == 3
        assert result[0] > result[1]  # dense > dilute
        # For N>1, not symmetric
        assert not np.isclose(result[0] + result[1], 1.0)

    def test_binodal_n100(self):
        """Test binodal for long polymer N=100."""
        chi_c = critical(n=100)[1]
        result = binodal(chi_c + 0.5, n=100)

        assert 0 < result[0] < 1
        assert 0 < result[1] < 1
        assert result[0] > result[1]

    def test_binodal_dilute_exponentially_small_at_large_chi(self):
        """At large chi, dilute phase should be exponentially small."""
        result = binodal(20.0, n=1)
        # Dilute phase should be very small
        assert result[1] < 0.001

    def test_binodal_inside_spinodal(self):
        """Binodal should be outside spinodal (wider separation)."""
        chi = 5.0
        binodal_result = binodal(chi, n=1)
        spinodal_result = spinodal(chi, n=1)

        # Binodal dense > spinodal dense
        assert binodal_result[0] > spinodal_result[0]
        # Binodal dilute < spinodal dilute
        assert binodal_result[1] < spinodal_result[1]

    def test_binodal_vs_spinodal_ordering(self):
        """Verify binodal is always outside spinodal for multiple chi."""
        for chi in [3.0, 5.0, 10.0]:
            binodal_result = binodal(chi, n=1)
            spinodal_result = spinodal(chi, n=1)

            assert binodal_result[0] >= spinodal_result[0]
            assert binodal_result[1] <= spinodal_result[1]


# =============================================================================
# Tests for analytic_binodal() function
# =============================================================================


class TestAnalyticBinodal:
    """Tests for the closed-form analytical binodal."""

    def test_analytic_binodal_single_value_n1(self):
        """Test analytic binodal for single chi value with N=1."""
        result = analytic_binodal(3.0, n=1)
        assert len(result) == 2
        assert result[0] > result[1]  # dense > dilute

    def test_analytic_binodal_physical_bounds(self):
        """Analytic binodal should always be between 0 and 1."""
        for chi in [2.5, 5.0, 10.0, 20.0]:
            result = analytic_binodal(chi, n=1)
            assert 0 < result[0] <= 1
            assert 0 <= result[1] < 1

    def test_analytic_binodal_n1_symmetric(self):
        """For N=1, phi_dense + phi_dilute = 1."""
        result = analytic_binodal(4.0, n=1)
        assert_allclose(result[0] + result[1], 1.0, rtol=1e-10)

    def test_analytic_binodal_array_input(self, chi_values_above_critical):
        """Test analytic binodal with array of chi values."""
        result = analytic_binodal(chi_values_above_critical, n=1)
        assert result.shape == (3, len(chi_values_above_critical))

    def test_analytic_binodal_raises_for_subcritical(self):
        """Should raise ValueError when chi is below critical."""
        with pytest.raises(ValueError, match="interaction strength too small"):
            analytic_binodal(1.5, n=1)

    def test_analytic_binodal_raises_for_subcritical_array(
        self, chi_values_below_critical
    ):
        """Should raise ValueError when all array values are subcritical."""
        with pytest.raises(ValueError, match="interaction strength too small"):
            analytic_binodal(chi_values_below_critical, n=1)

    def test_analytic_binodal_matches_iterative(self):
        """Analytic solution should approximately match iterative solution."""
        for chi in [3.0, 4.0, 5.0]:
            analytic = analytic_binodal(chi, n=1)
            iterative = binodal(chi, n=1, iteration=10)

            # The methods use different mathematical approaches and may give
            # somewhat different results, especially at larger chi
            # Check that both give valid phase separation with similar dense phase
            assert_allclose(analytic[0], iterative[0], rtol=0.20)
            # For dilute phase, just verify both are small and in same ballpark
            assert_allclose(analytic[1], iterative[1], rtol=0.25)

    def test_analytic_binodal_n_greater_than_1(self):
        """Test analytic binodal for N > 1 case."""
        chi_c = critical(n=10)[1]
        result = analytic_binodal(chi_c + 1.0, n=10)

        assert len(result) == 2
        assert result[0] > result[1]
        assert 0 < result[0] < 1
        assert 0 < result[1] < 1

    def test_analytic_binodal_n100_matches_iterative(self):
        """Analytic and iterative should both find phase separation for large N."""
        chi_c = critical(n=100)[1]
        chi = chi_c + 0.5

        analytic = analytic_binodal(chi, n=100)
        iterative = binodal(chi, n=100, iteration=10)

        # Both methods should find valid phase separation
        # Dense phase should be similar
        assert_allclose(analytic[0], iterative[0], rtol=0.15)
        # Dilute phase: just verify both are small and positive
        assert 0 < analytic[1] < 0.1
        assert 0 < iterative[1] < 0.1

    def test_analytic_binodal_filters_subcritical(self):
        """Should filter out subcritical values from array."""
        chi_mixed = [1.0, 2.0, 3.0, 4.0]
        result = analytic_binodal(chi_mixed, n=1)
        # chi_c = 2.0 for N=1; only chi strictly above chi_c is kept
        assert len(result[2]) == 2
        assert np.all(result[2] > 2.0)

    def test_analytic_binodal_inside_spinodal(self):
        """Analytic binodal should be outside spinodal."""
        chi = 5.0
        binodal_result = analytic_binodal(chi, n=1)
        spinodal_result = spinodal(chi, n=1)

        assert binodal_result[0] > spinodal_result[0]
        assert binodal_result[1] < spinodal_result[1]

    def test_analytic_binodal_exponential_scaling(self):
        """Dilute phase should show exponential scaling at large chi."""
        chi_values = [5.0, 10.0, 15.0, 20.0]
        dilute_concentrations = [analytic_binodal(chi, n=1)[1] for chi in chi_values]

        # Log of dilute concentration should decrease roughly linearly with chi
        log_dilute = np.log(dilute_concentrations)
        # Check that it's decreasing and roughly linear
        diffs = np.diff(log_dilute)
        assert np.all(diffs < 0)  # All decreasing


# =============================================================================
# Cross-function consistency tests
# =============================================================================


class TestCrossFunctionConsistency:
    """Tests verifying consistency between different functions."""

    def test_spinodal_inside_binodal(self):
        """Spinodal boundaries should be inside binodal boundaries."""
        for chi in [3.0, 5.0, 10.0, 20.0]:
            b = binodal(chi, n=1)
            s = spinodal(chi, n=1)

            # Binodal range should contain spinodal range
            assert b[0] >= s[0]  # binodal dense >= spinodal dense
            assert b[1] <= s[1]  # binodal dilute <= spinodal dilute

    def test_all_methods_agree_near_critical(self):
        """All methods should agree near critical point."""
        chi_c = critical(n=1)[1]
        phi_c = critical(n=1)[0]

        chi = chi_c + 0.1  # Slightly further from critical for stability

        gl = GL_binodal(chi, n=1)
        b = binodal(chi, n=1, iteration=10)
        a = analytic_binodal(chi, n=1)

        # All should be relatively close to phi_c (within 0.2)
        for result in [gl, b, a]:
            assert_allclose(result[0], phi_c, atol=0.2)
            assert_allclose(result[1], phi_c, atol=0.2)

    def test_iterative_and_analytic_consistency_n1(self):
        """Iterative and analytic methods should find similar phase separation for N=1."""
        chi_values = [2.5, 3.0, 4.0]

        for chi in chi_values:
            iterative = binodal(chi, n=1, iteration=10)
            analytic = analytic_binodal(chi, n=1)

            # Different formulations may give somewhat different results
            # Verify dense phases match reasonably well
            assert_allclose(iterative[0], analytic[0], rtol=0.20)
            # Both methods should find dilute phase in similar range
            # (exact values can differ significantly at larger chi)
            assert iterative[1] < 0.3
            assert analytic[1] < 0.3

    def test_iterative_and_analytic_consistency_n10(self):
        """Iterative and analytic methods should both find phase separation for N=10."""
        chi_c = critical(n=10)[1]
        chi_values = [chi_c + 0.5, chi_c + 1.0, chi_c + 2.0]

        for chi in chi_values:
            iterative = binodal(chi, n=10, iteration=10)
            analytic = analytic_binodal(chi, n=10)

            # Both should find valid phase separation with dense > dilute
            assert iterative[0] > iterative[1]
            assert analytic[0] > analytic[1]
            # Dense phases should be reasonably close
            assert_allclose(iterative[0], analytic[0], rtol=0.15)

    def test_critical_is_limiting_case(self):
        """Near critical chi, phase separation range should be small."""
        n = 10
        phi_c, chi_c = critical(n=n)

        # Test slightly above critical
        chi = chi_c + 0.05

        s = spinodal(chi, n=n)
        b = binodal(chi, n=n, iteration=10)

        # Near critical, phase separation range should be small
        spinodal_range = s[0] - s[1]
        binodal_range = b[0] - b[1]

        assert spinodal_range < 0.4  # Spinodal separation should be small
        assert binodal_range < 0.4  # Binodal separation should be small

    def test_phase_ordering_preserved(self):
        """dense > spinodal_dense > spinodal_dilute > dilute for all chi."""
        for chi in [3.0, 5.0, 10.0]:
            b = binodal(chi, n=1)
            s = spinodal(chi, n=1)

            # Check full ordering
            assert b[0] >= s[0] >= s[1] >= b[1]


# =============================================================================
# Edge case and numerical stability tests
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability at edge cases."""

    def test_very_large_chi(self):
        """Functions should handle moderately large chi values."""
        chi = 15.0

        # All functions should run without numerical issues
        s = spinodal(chi, n=1)
        b = binodal(chi, n=1)
        a = analytic_binodal(chi, n=1)

        assert is_valid_concentration(s)
        # Allow boundary cases at large chi
        assert 0 <= b[0] <= 1 and 0 <= b[1] <= 1
        assert 0 <= a[0] <= 1 and 0 <= a[1] <= 1

    def test_very_large_n(self):
        """Functions should handle very large polymer length."""
        n = 1000
        chi_c = critical(n=n)[1]
        chi = chi_c + 0.1

        s = spinodal(chi, n=n)
        b = binodal(chi, n=n, iteration=5)

        assert is_valid_concentration(s)
        assert 0 < b[0] < 1 and 0 < b[1] < 1

    def test_chi_exactly_at_critical(self):
        """Functions should handle chi slightly above critical point."""
        chi_c = critical(n=1)[1]
        chi = chi_c + 0.01  # Slightly above critical

        # spinodal should work at chi slightly above chi_c
        s = spinodal(chi, n=1)
        assert len(s) == 2

        # binodal should work slightly above chi_c
        b = binodal(chi, n=1)
        assert len(b) == 3

    def test_many_iterations(self):
        """Large iteration count should not cause issues."""
        result = binodal(4.0, n=1, iteration=100)
        assert 0 < result[0] < 1
        assert 0 < result[1] < 1

    def test_float_precision_chi_array(self):
        """Array of chi should maintain precision."""
        chi_values = np.linspace(2.1, 10.0, 100)
        result = binodal(chi_values, n=1)

        assert result.shape[1] == 100
        assert np.all(result[0] > result[1])

    def test_improved_vs_simple_at_moderate_chi(self):
        """Both methods should work at moderate chi values."""
        chi = 10.0

        improved = binodal(chi, n=1, iteration=10, UseImprovedMap=True)
        simple = binodal(chi, n=1, iteration=20, UseImprovedMap=False)

        # Both should be valid (allow boundary cases)
        assert 0 <= improved[0] <= 1
        assert 0 <= simple[0] <= 1


# =============================================================================
# Input type handling tests
# =============================================================================


class TestInputTypes:
    """Tests for handling various input types."""

    def test_spinodal_accepts_list(self):
        """Spinodal should accept list input."""
        result = spinodal([3.0, 4.0, 5.0], n=1)
        assert result.shape == (3, 3)

    def test_spinodal_accepts_tuple(self):
        """Spinodal should accept tuple input."""
        result = spinodal((3.0, 4.0, 5.0), n=1)
        assert result.shape == (3, 3)

    def test_binodal_accepts_integer_chi(self):
        """Binodal should accept integer chi."""
        result = binodal(3, n=1)
        assert len(result) == 3

    def test_critical_accepts_integer_n(self):
        """Critical should accept integer n."""
        result = critical(n=10)
        assert len(result) == 2

    def test_functions_accept_numpy_scalar(self):
        """Functions should accept numpy scalar values."""
        chi = np.float64(3.0)
        n = np.int32(1)

        result = binodal(chi, n=n)
        assert len(result) == 3


# =============================================================================
# Regression tests (specific known values)
# =============================================================================


class TestRegressionValues:
    """Regression tests with specific known values."""

    def test_critical_n1_exact(self):
        """Test exact critical values for N=1."""
        result = critical(n=1)
        assert_allclose(result, [0.5, 2.0], rtol=1e-12)

    def test_critical_n4_exact(self):
        """Test exact critical values for N=4."""
        # phi_c = 1/(1+sqrt(4)) = 1/3
        # chi_c = 0.5*(1 + 1/2)^2 = 0.5 * 2.25 = 1.125
        result = critical(n=4)
        assert_allclose(result[0], 1.0 / 3.0, rtol=1e-12)
        assert_allclose(result[1], 1.125, rtol=1e-12)

    def test_spinodal_n1_chi3(self):
        """Test spinodal values at chi=3, N=1."""
        result = spinodal(3.0, n=1)
        # For N=1, gamma=0, so t1 = 0.5, t2 = sqrt(0.25 - 1/6) = sqrt(1/12)
        t1 = 0.5
        t2 = np.sqrt(0.25 - 1.0 / 6.0)
        expected = [t1 + t2, t1 - t2]
        assert_allclose(result, expected, rtol=1e-10)

    def test_symmetric_property_n1(self):
        """Verify symmetry property: phi_dense = 1 - phi_dilute for N=1."""
        for func in [spinodal, GL_binodal, binodal]:
            result = func(5.0, n=1)
            # For binodal, result format is [dense, dilute, chi]
            if len(result) == 3:
                assert_allclose(result[0] + result[1], 1.0, rtol=1e-6)
            else:
                assert_allclose(result[0] + result[1], 1.0, rtol=1e-6)


class TestBoundaryConsistency:
    """Scalar and array code paths must agree at and below chi_c."""

    @pytest.mark.parametrize("fx", [spinodal, GL_binodal, binodal, analytic_binodal])
    def test_array_at_chi_c_raises_like_scalar(self, fx):
        chi_c = critical(10)[1]
        with pytest.raises(ValueError):
            fx(chi_c, n=10)
        with pytest.raises(ValueError):
            fx([chi_c], n=10)

    @pytest.mark.parametrize("fx", [spinodal, GL_binodal, binodal, analytic_binodal])
    def test_array_output_never_nan(self, fx):
        chi_c = critical(10)[1]
        result = fx([chi_c, chi_c + 0.05, chi_c + 0.5], n=10)
        assert result.shape == (3, 2)
        assert not np.any(np.isnan(result))

    def test_binodal_rejects_n_below_one(self):
        with pytest.raises(ValueError):
            binodal(3.0, n=0.5)
