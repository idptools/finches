"""
Tests for frontend_base.py module.

Tests the FinchesFrontend base class functionality through the Mpipi_frontend
derived class, since the base class should not be instantiated directly.

Tests cover:
- Base class instantiation prevention
- intermolecular_idr_matrix function
- epsilon function
- epsilon_vectors function
- interaction_figure function
- per_residue_attractive_vector function
- per_residue_repulsive_vector function
- protein_nucleic_vector function
- plot_protein_nucleic_vector function
- build_phase_diagram function
- plot_phase_diagram function
- plot_multiple_phase_diagrams function
"""

import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from finches import Mpipi_frontend
from finches.frontend.frontend_base import FinchesFrontend


# Change to test directory for any data files
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# TEST SEQUENCES
# =============================================================================

# Short test sequences for quick tests
SHORT_SEQ_1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVS"  # 30 aa
SHORT_SEQ_2 = "AEKLSQPGWYFVMNHAEKLSQPGWYFVMNH"  # 30 aa

# Medium test sequences
MEDIUM_SEQ_1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTG"  # 50 aa
MEDIUM_SEQ_2 = "GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGG"  # 50 aa

# Longer sequences for matrix tests (need to be longer than window_size)
LONG_SEQ_1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
LONG_SEQ_2 = "GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSFSSSYGSGRRFGGGSGGGGGSSSSSGGGGRGSGGGRGGGGSFGGGRGGGSFGGGRGGGGGGGSFGGGGRGGGGSGGGGFRGRGRGRGRGRGRGRGRGRGRGRGRGRG"

# Charged sequences
POSITIVE_SEQ = "KRKRKRKRKRKRKRKRKRKRKRKRKRKRKRKRKRKRKRKRKRKRKRKRKR"  # 50 aa
NEGATIVE_SEQ = "EDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDEDE"  # 50 aa

# IDP-like sequence (for disorder testing)
IDP_SEQ = "GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSFSSSYGSGRRF"

# RNA sequence (polyU)
RNA_SEQ = "UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU"


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mf():
    """Create a Mpipi_frontend instance for testing."""
    return Mpipi_frontend()


@pytest.fixture
def mf_custom_salt():
    """Create a Mpipi_frontend instance with custom salt."""
    return Mpipi_frontend(salt=0.100)


# =============================================================================
# BASE CLASS TESTS
# =============================================================================

class TestFinchesFrontendBase:
    """Tests for the FinchesFrontend base class behavior."""
    
    def test_base_class_cannot_be_instantiated(self):
        """Test that FinchesFrontend cannot be instantiated directly."""
        with pytest.raises(TypeError) as excinfo:
            FinchesFrontend()
        assert "FinchesFrontend class should not be instantiated directly" in str(excinfo.value)
    
    def test_derived_class_can_be_instantiated(self, mf):
        """Test that derived classes can be instantiated."""
        assert mf is not None
        assert isinstance(mf, FinchesFrontend)
        assert mf.IMC_object is not None


# =============================================================================
# EPSILON TESTS
# =============================================================================

class TestEpsilon:
    """Tests for the epsilon() function."""
    
    def test_epsilon_returns_float(self, mf):
        """Test that epsilon returns a float value."""
        result = mf.epsilon(SHORT_SEQ_1, SHORT_SEQ_2)
        assert isinstance(result, (int, float, np.floating))
    
    def test_epsilon_homotypic(self, mf):
        """Test epsilon for homotypic (same sequence) interactions."""
        result = mf.epsilon(SHORT_SEQ_1, SHORT_SEQ_1)
        assert isinstance(result, (int, float, np.floating))
    
    def test_epsilon_heterotypic(self, mf):
        """Test epsilon for heterotypic (different sequence) interactions."""
        result = mf.epsilon(SHORT_SEQ_1, SHORT_SEQ_2)
        assert isinstance(result, (int, float, np.floating))
    
    def test_epsilon_symmetric(self, mf):
        """Test that epsilon is symmetric: epsilon(A,B) == epsilon(B,A)."""
        eps_ab = mf.epsilon(SHORT_SEQ_1, SHORT_SEQ_2)
        eps_ba = mf.epsilon(SHORT_SEQ_2, SHORT_SEQ_1)
        assert np.isclose(eps_ab, eps_ba, rtol=1e-10)
    
    def test_epsilon_charged_sequences(self, mf):
        """Test epsilon with charged sequences."""
        # Opposite charges should be more attractive (more negative)
        eps_same_pos = mf.epsilon(POSITIVE_SEQ, POSITIVE_SEQ)
        eps_same_neg = mf.epsilon(NEGATIVE_SEQ, NEGATIVE_SEQ)
        eps_opposite = mf.epsilon(POSITIVE_SEQ, NEGATIVE_SEQ)
        
        # Opposite charges should be more attractive than like charges
        assert eps_opposite < eps_same_pos
        assert eps_opposite < eps_same_neg
    
    def test_epsilon_weighting_options(self, mf):
        """Test epsilon with different weighting options."""
        eps_default = mf.epsilon(SHORT_SEQ_1, SHORT_SEQ_2)
        eps_no_aliphatic = mf.epsilon(SHORT_SEQ_1, SHORT_SEQ_2, use_aliphatic_weighting=False)
        eps_no_charge = mf.epsilon(SHORT_SEQ_1, SHORT_SEQ_2, use_charge_weighting=False)
        eps_no_weights = mf.epsilon(SHORT_SEQ_1, SHORT_SEQ_2, 
                                    use_aliphatic_weighting=False, 
                                    use_charge_weighting=False)
        
        # All should return valid floats
        assert all(isinstance(x, (int, float, np.floating)) 
                   for x in [eps_default, eps_no_aliphatic, eps_no_charge, eps_no_weights])


# =============================================================================
# EPSILON VECTORS TESTS
# =============================================================================

class TestEpsilonVectors:
    """Tests for the epsilon_vectors() function."""
    
    def test_epsilon_vectors_returns_tuple(self, mf):
        """Test that epsilon_vectors returns a tuple."""
        result = mf.epsilon_vectors(SHORT_SEQ_1, SHORT_SEQ_2)
        assert isinstance(result, tuple)
    
    def test_epsilon_vectors_weighting_options(self, mf):
        """Test epsilon_vectors with different weighting options."""
        result_default = mf.epsilon_vectors(SHORT_SEQ_1, SHORT_SEQ_2)
        result_no_aliphatic = mf.epsilon_vectors(SHORT_SEQ_1, SHORT_SEQ_2, 
                                                  use_aliphatic_weighting=False)
        result_no_charge = mf.epsilon_vectors(SHORT_SEQ_1, SHORT_SEQ_2, 
                                               use_charge_weighting=False)
        
        # All should return tuples
        assert all(isinstance(x, tuple) for x in [result_default, result_no_aliphatic, result_no_charge])


# =============================================================================
# INTERMOLECULAR IDR MATRIX TESTS
# =============================================================================

class TestIntermolecularIdrMatrix:
    """Tests for the intermolecular_idr_matrix() function."""
    
    def test_matrix_returns_tuple(self, mf):
        """Test that intermolecular_idr_matrix returns a tuple."""
        result = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_matrix_structure(self, mf):
        """Test the structure of the returned matrix tuple."""
        result = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        
        # B is a tuple of (matrix, indices1, indices2)
        B = result[0]
        disorder_1 = result[1]
        disorder_2 = result[2]
        
        assert isinstance(B, tuple)
        assert len(B) == 3
        
        # Matrix should be 2D numpy array
        assert isinstance(B[0], np.ndarray)
        assert len(B[0].shape) == 2
        
        # Indices should be arrays
        assert isinstance(B[1], np.ndarray)
        assert isinstance(B[2], np.ndarray)
        
        # Disorder profiles should be arrays
        assert isinstance(disorder_1, np.ndarray)
        assert isinstance(disorder_2, np.ndarray)
    
    def test_matrix_dimensions_match_indices(self, mf):
        """Test that matrix dimensions match the index arrays."""
        result = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        B = result[0]
        
        # Matrix shape should match index lengths
        assert B[0].shape[0] == len(B[1])
        assert B[0].shape[1] == len(B[2])
    
    def test_matrix_disorder_dimensions(self, mf):
        """Test that disorder profiles match matrix dimensions."""
        result = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        B = result[0]
        disorder_1 = result[1]
        disorder_2 = result[2]
        
        assert len(disorder_1) == B[0].shape[0]
        assert len(disorder_2) == B[0].shape[1]
    
    def test_matrix_window_size_effect(self, mf):
        """Test that different window sizes produce different sized matrices."""
        result_31 = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        result_21 = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, window_size=21)
        
        # Smaller window should give larger matrix
        assert result_21[0][0].shape[0] > result_31[0][0].shape[0]
        assert result_21[0][0].shape[1] > result_31[0][0].shape[1]
    
    def test_matrix_homotypic(self, mf):
        """Test matrix calculation for homotypic interactions."""
        result = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_1, window_size=31)
        B = result[0]
        
        # Homotypic matrix should be square
        assert B[0].shape[0] == B[0].shape[1]
        
        # Homotypic matrix should be symmetric
        assert np.allclose(B[0], B[0].T, rtol=1e-10)
    
    def test_matrix_disorder_disabled(self, mf):
        """Test matrix with disorder profiles disabled."""
        result = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, 
                                               window_size=31,
                                               disorder_1=False, 
                                               disorder_2=False)
        disorder_1 = result[1]
        disorder_2 = result[2]
        
        # Disabled disorder should return all 1s
        assert np.all(disorder_1 == 1)
        assert np.all(disorder_2 == 1)
    
    def test_matrix_weighting_options(self, mf):
        """Test matrix with different weighting options."""
        result_default = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        result_no_aliphatic = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, 
                                                            window_size=31,
                                                            use_aliphatic_weighting=False)
        result_no_charge = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, 
                                                         window_size=31,
                                                         use_charge_weighting=False)
        
        # Results should be different with different weightings
        assert not np.allclose(result_default[0][0], result_no_aliphatic[0][0])
        assert not np.allclose(result_default[0][0], result_no_charge[0][0])
    
    def test_matrix_null_shuffle_invalid_input(self, mf):
        """Test that null_shuffle raises error for invalid input."""
        with pytest.raises(ValueError):
            mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, 
                                         window_size=31, 
                                         null_shuffle="invalid")
    
    def test_matrix_null_shuffle_bool_rejected(self, mf):
        """Test that null_shuffle=True raises error (must be int/float, not bool)."""
        with pytest.raises(ValueError):
            mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, 
                                         window_size=31, 
                                         null_shuffle=True)
    
    def test_matrix_null_shuffle_small(self, mf):
        """Test matrix with small null_shuffle value."""
        result = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, 
                                               window_size=31, 
                                               null_shuffle=5)
        # Should still return valid structure
        assert isinstance(result, tuple)
        assert len(result) == 3


# =============================================================================
# INTERACTION FIGURE TESTS
# =============================================================================

class TestInteractionFigure:
    """Tests for the interaction_figure() function."""
    
    def test_figure_returns_tuple(self, mf):
        """Test that interaction_figure returns a tuple."""
        result = mf.interaction_figure(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        assert isinstance(result, tuple)
        plt.close('all')
    
    def test_figure_with_disorder(self, mf):
        """Test figure with disorder profiles enabled."""
        result = mf.interaction_figure(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        
        # Should return 6 elements when disorder is shown
        assert len(result) == 6
        
        fig, im, ax_main, ax_top, ax_right, ax_colorbar = result
        assert fig is not None
        assert im is not None
        assert ax_main is not None
        assert ax_top is not None
        assert ax_right is not None
        assert ax_colorbar is not None
        plt.close('all')
    
    def test_figure_without_disorder(self, mf):
        """Test figure with disorder profiles disabled."""
        result = mf.interaction_figure(LONG_SEQ_1, LONG_SEQ_2, 
                                        window_size=31, 
                                        no_disorder=True)
        
        # Should return 3 elements when disorder is not shown
        assert len(result) == 3
        
        fig, im, ax_main = result
        assert fig is not None
        assert im is not None
        assert ax_main is not None
        plt.close('all')
    
    def test_figure_custom_colormap(self, mf):
        """Test figure with custom colormap."""
        result = mf.interaction_figure(LONG_SEQ_1, LONG_SEQ_2, 
                                        window_size=31, 
                                        cmap='coolwarm')
        assert result is not None
        plt.close('all')
    
    def test_figure_custom_vmin_vmax(self, mf):
        """Test figure with custom vmin/vmax."""
        result = mf.interaction_figure(LONG_SEQ_1, LONG_SEQ_2, 
                                        window_size=31, 
                                        vmin=-5, vmax=5)
        assert result is not None
        plt.close('all')
    
    def test_figure_with_domains(self, mf):
        """Test figure with domain annotations."""
        seq1_domains = [(10, 30), (50, 70)]
        seq2_domains = [(20, 40)]
        
        result = mf.interaction_figure(LONG_SEQ_1, LONG_SEQ_2, 
                                        window_size=31,
                                        seq1_domains=seq1_domains,
                                        seq2_domains=seq2_domains)
        assert result is not None
        plt.close('all')
    
    def test_figure_with_lines(self, mf):
        """Test figure with line annotations."""
        result = mf.interaction_figure(LONG_SEQ_1, LONG_SEQ_2, 
                                        window_size=31,
                                        seq1_lines=[50, 100],
                                        seq2_lines=[30, 60])
        assert result is not None
        plt.close('all')
    
    def test_figure_with_rectangles(self, mf):
        """Test figure with rectangle annotations."""
        rectangles = [
            [20, 40, 30, 50, {'edgecolor': 'red'}],
        ]
        result = mf.interaction_figure(LONG_SEQ_1, LONG_SEQ_2, 
                                        window_size=31,
                                        plot_rectangles=rectangles)
        assert result is not None
        plt.close('all')
    
    def test_figure_save_to_file(self, mf, tmp_path):
        """Test saving figure to file."""
        fname = tmp_path / "test_figure.png"
        result = mf.interaction_figure(LONG_SEQ_1, LONG_SEQ_2, 
                                        window_size=31,
                                        fname=str(fname))
        assert fname.exists()
        plt.close('all')
    
    def test_figure_zero_folded(self, mf):
        """Test figure with zero_folded option."""
        result_zero = mf.interaction_figure(LONG_SEQ_1, LONG_SEQ_2, 
                                             window_size=31,
                                             zero_folded=True)
        result_no_zero = mf.interaction_figure(LONG_SEQ_1, LONG_SEQ_2, 
                                                window_size=31,
                                                zero_folded=False)
        assert result_zero is not None
        assert result_no_zero is not None
        plt.close('all')


# =============================================================================
# PER RESIDUE ATTRACTIVE VECTOR TESTS
# =============================================================================

class TestPerResidueAttractiveVector:
    """Tests for the per_residue_attractive_vector() function."""
    
    def test_attractive_vector_returns_tuple(self, mf):
        """Test that per_residue_attractive_vector returns a tuple."""
        result = mf.per_residue_attractive_vector(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_attractive_vector_structure(self, mf):
        """Test the structure of returned arrays."""
        idx, vals = mf.per_residue_attractive_vector(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        
        assert isinstance(idx, np.ndarray)
        assert isinstance(vals, np.ndarray)
        assert len(idx) == len(vals)
    
    def test_attractive_vector_values_negative(self, mf):
        """Test that attractive values are typically negative or zero."""
        idx, vals = mf.per_residue_attractive_vector(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        
        # Attractive values should be <= 0 (or close due to smoothing)
        # Note: smoothing may cause some values to be slightly positive
        assert np.mean(vals) <= 0.5  # Allow some tolerance for smoothing
    
    def test_attractive_vector_return_total(self, mf):
        """Test return_total option."""
        idx1, vals1 = mf.per_residue_attractive_vector(LONG_SEQ_1, LONG_SEQ_2, 
                                                        window_size=31,
                                                        return_total=False)
        idx2, vals2 = mf.per_residue_attractive_vector(LONG_SEQ_1, LONG_SEQ_2, 
                                                        window_size=31,
                                                        return_total=True)
        
        # Indices should be the same
        assert np.array_equal(idx1, idx2)
        
        # Total should generally have larger magnitude
        # (though this depends on the sequences)
        assert not np.allclose(vals1, vals2)
    
    def test_attractive_vector_no_smoothing(self, mf):
        """Test without smoothing."""
        idx, vals = mf.per_residue_attractive_vector(LONG_SEQ_1, LONG_SEQ_2, 
                                                      window_size=31,
                                                      smoothing_window=False)
        assert isinstance(vals, np.ndarray)
    
    def test_attractive_vector_custom_threshold(self, mf):
        """Test with custom attractive threshold."""
        idx1, vals1 = mf.per_residue_attractive_vector(LONG_SEQ_1, LONG_SEQ_2, 
                                                        window_size=31,
                                                        attractive_threshold=0)
        idx2, vals2 = mf.per_residue_attractive_vector(LONG_SEQ_1, LONG_SEQ_2, 
                                                        window_size=31,
                                                        attractive_threshold=-1)
        
        # Different thresholds should give different results
        assert not np.allclose(vals1, vals2)


# =============================================================================
# PER RESIDUE REPULSIVE VECTOR TESTS
# =============================================================================

class TestPerResidueRepulsiveVector:
    """Tests for the per_residue_repulsive_vector() function."""
    
    def test_repulsive_vector_returns_tuple(self, mf):
        """Test that per_residue_repulsive_vector returns a tuple."""
        result = mf.per_residue_repulsive_vector(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_repulsive_vector_structure(self, mf):
        """Test the structure of returned arrays."""
        idx, vals = mf.per_residue_repulsive_vector(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        
        assert isinstance(idx, np.ndarray)
        assert isinstance(vals, np.ndarray)
        assert len(idx) == len(vals)
    
    def test_repulsive_vector_values_positive(self, mf):
        """Test that repulsive values are typically positive or zero."""
        idx, vals = mf.per_residue_repulsive_vector(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        
        # Repulsive values should be >= 0 (or close due to smoothing)
        assert np.mean(vals) >= -0.5  # Allow some tolerance for smoothing
    
    def test_repulsive_vector_return_total(self, mf):
        """Test return_total option."""
        idx1, vals1 = mf.per_residue_repulsive_vector(LONG_SEQ_1, LONG_SEQ_2, 
                                                       window_size=31,
                                                       return_total=False)
        idx2, vals2 = mf.per_residue_repulsive_vector(LONG_SEQ_1, LONG_SEQ_2, 
                                                       window_size=31,
                                                       return_total=True)
        
        # Indices should be the same
        assert np.array_equal(idx1, idx2)
    
    def test_repulsive_vector_no_smoothing(self, mf):
        """Test without smoothing."""
        idx, vals = mf.per_residue_repulsive_vector(LONG_SEQ_1, LONG_SEQ_2, 
                                                     window_size=31,
                                                     smoothing_window=False)
        assert isinstance(vals, np.ndarray)
    
    def test_repulsive_vector_custom_threshold(self, mf):
        """Test with custom repulsive threshold."""
        idx1, vals1 = mf.per_residue_repulsive_vector(LONG_SEQ_1, LONG_SEQ_2, 
                                                       window_size=31,
                                                       repulsive_threshold=0)
        idx2, vals2 = mf.per_residue_repulsive_vector(LONG_SEQ_1, LONG_SEQ_2, 
                                                       window_size=31,
                                                       repulsive_threshold=1)
        
        # Different thresholds should give different results
        assert not np.allclose(vals1, vals2)


# =============================================================================
# PROTEIN NUCLEIC VECTOR TESTS
# =============================================================================

class TestProteinNucleicVector:
    """Tests for the protein_nucleic_vector() function."""
    
    def test_nucleic_vector_returns_list(self, mf):
        """Test that protein_nucleic_vector returns a list."""
        result = mf.protein_nucleic_vector(LONG_SEQ_1, fragsize=21)
        assert isinstance(result, list)
        assert len(result) == 2
    
    def test_nucleic_vector_structure(self, mf):
        """Test the structure of returned arrays."""
        idx, vals = mf.protein_nucleic_vector(LONG_SEQ_1, fragsize=21)
        
        assert isinstance(idx, np.ndarray)
        assert isinstance(vals, np.ndarray)
        assert len(idx) == len(vals)
    
    def test_nucleic_vector_fragsize_must_be_odd(self, mf):
        """Test that fragsize must be odd."""
        with pytest.raises(Exception):
            mf.protein_nucleic_vector(LONG_SEQ_1, fragsize=20)
    
    def test_nucleic_vector_no_smoothing(self, mf):
        """Test without smoothing."""
        result = mf.protein_nucleic_vector(LONG_SEQ_1, fragsize=21, 
                                            smoothing_window=False)
        assert isinstance(result, list)
    
    def test_nucleic_vector_short_sequence(self, mf):
        """Test with sequence shorter than fragsize."""
        result = mf.protein_nucleic_vector(SHORT_SEQ_1, fragsize=31, 
                                            smoothing_window=False)
        
        # Should still return valid structure
        idx, vals = result
        assert len(idx) == 1  # Single value for short sequence
        assert len(vals) == 1
    
    def test_nucleic_vector_charged_sequences(self, mf):
        """Test that positive sequences show RNA binding."""
        idx_pos, vals_pos = mf.protein_nucleic_vector(POSITIVE_SEQ, fragsize=21, 
                                                       smoothing_window=False)
        idx_neg, vals_neg = mf.protein_nucleic_vector(NEGATIVE_SEQ, fragsize=21, 
                                                       smoothing_window=False)
        
        # Positive sequences should bind RNA more strongly (more negative epsilon)
        assert np.mean(vals_pos) < np.mean(vals_neg)


# =============================================================================
# PLOT PROTEIN NUCLEIC VECTOR TESTS
# =============================================================================

class TestPlotProteinNucleicVector:
    """Tests for the plot_protein_nucleic_vector() function."""
    
    def test_plot_nucleic_returns_tuple(self, mf):
        """Test that plot_protein_nucleic_vector returns figure and axes."""
        result = mf.plot_protein_nucleic_vector(LONG_SEQ_1, fragsize=21)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        fig, ax = result
        assert fig is not None
        assert ax is not None
        plt.close('all')
    
    def test_plot_nucleic_with_domains(self, mf):
        """Test plot with domain annotations."""
        result = mf.plot_protein_nucleic_vector(LONG_SEQ_1, fragsize=21,
                                                 domains=[(10, 50), (100, 150)])
        assert result is not None
        plt.close('all')
    
    def test_plot_nucleic_custom_colors(self, mf):
        """Test plot with custom colors."""
        result = mf.plot_protein_nucleic_vector(LONG_SEQ_1, fragsize=21,
                                                 domain_color='blue',
                                                 domain_alpha=0.5)
        assert result is not None
        plt.close('all')
    
    def test_plot_nucleic_save_to_file(self, mf, tmp_path):
        """Test saving plot to file."""
        fname = tmp_path / "test_nucleic.png"
        result = mf.plot_protein_nucleic_vector(LONG_SEQ_1, fragsize=21,
                                                 fname=str(fname))
        assert fname.exists()
        plt.close('all')
    
    def test_plot_nucleic_custom_ylim(self, mf):
        """Test plot with custom y limits."""
        result = mf.plot_protein_nucleic_vector(LONG_SEQ_1, fragsize=21,
                                                 ylim=[-2, 2])
        assert result is not None
        plt.close('all')
    
    def test_plot_nucleic_with_grid(self, mf):
        """Test plot with grid enabled."""
        result = mf.plot_protein_nucleic_vector(LONG_SEQ_1, fragsize=21,
                                                 show_grid=True)
        assert result is not None
        plt.close('all')


# =============================================================================
# PROTEIN PEPTIDE VECTOR TESTS
# =============================================================================

# Test peptides
PEPTIDE_AROMATIC = "FYWFYW"
PEPTIDE_CHARGED = "KRKRKR"
PEPTIDE_SHORT = "FYW"


class TestProteinPeptideVector:
    """Tests for the protein_peptide_vector() function."""
    
    def test_peptide_vector_returns_list(self, mf):
        """Test that protein_peptide_vector returns a list."""
        result = mf.protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21)
        assert isinstance(result, list)
        assert len(result) == 2
    
    def test_peptide_vector_structure(self, mf):
        """Test the structure of returned arrays."""
        idx, vals = mf.protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21)
        
        assert isinstance(idx, np.ndarray)
        assert isinstance(vals, np.ndarray)
        assert len(idx) == len(vals)
    
    def test_peptide_vector_fragsize_must_be_odd(self, mf):
        """Test that fragsize must be odd when explicitly provided."""
        with pytest.raises(Exception):
            mf.protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=20)
    
    def test_peptide_vector_default_fragsize(self, mf):
        """Test that default fragsize is 21."""
        # Should work without specifying fragsize (defaults to 21)
        result = mf.protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC)
        assert isinstance(result, list)
    
    def test_peptide_vector_fragsize_none_raises(self, mf):
        """Test that passing fragsize=None raises an exception."""
        with pytest.raises(Exception):
            mf.protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=None)
    
    def test_peptide_vector_no_smoothing(self, mf):
        """Test without smoothing."""
        result = mf.protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21, 
                                            smoothing_window=False)
        assert isinstance(result, list)
    
    def test_peptide_vector_short_sequence(self, mf):
        """Test with sequence shorter than fragsize."""
        result = mf.protein_peptide_vector(SHORT_SEQ_1, PEPTIDE_AROMATIC, fragsize=31, 
                                            smoothing_window=False)
        
        # Should still return valid structure
        idx, vals = result
        assert len(idx) == 1  # Single value for short sequence
        assert len(vals) == 1
    
    def test_peptide_vector_different_peptides(self, mf):
        """Test that different peptides give different binding profiles."""
        idx_arom, vals_arom = mf.protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, 
                                                         fragsize=21, smoothing_window=False)
        idx_chg, vals_chg = mf.protein_peptide_vector(LONG_SEQ_1, PEPTIDE_CHARGED, 
                                                       fragsize=21, smoothing_window=False)
        
        # Different peptides should give different results
        assert not np.allclose(vals_arom, vals_chg)
    
    def test_peptide_vector_short_peptide(self, mf):
        """Test with a short peptide."""
        result = mf.protein_peptide_vector(LONG_SEQ_1, PEPTIDE_SHORT, fragsize=21)
        idx, vals = result
        assert len(idx) == len(vals)
    
    def test_peptide_vector_custom_smoothing(self, mf):
        """Test with custom smoothing parameters."""
        result = mf.protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, 
                                            fragsize=21,
                                            smoothing_window=15,
                                            poly_order=2)
        assert isinstance(result, list)


# =============================================================================
# PLOT PROTEIN PEPTIDE VECTOR TESTS
# =============================================================================

class TestPlotProteinPeptideVector:
    """Tests for the plot_protein_peptide_vector() function."""
    
    def test_plot_peptide_returns_tuple(self, mf):
        """Test that plot_protein_peptide_vector returns figure and axes."""
        result = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        fig, ax = result
        assert fig is not None
        assert ax is not None
        plt.close('all')
    
    def test_plot_peptide_default_fragsize(self, mf):
        """Test plot with default fragsize (21)."""
        result = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC)
        assert result is not None
        plt.close('all')
    
    def test_plot_peptide_with_domains(self, mf):
        """Test plot with domain annotations."""
        result = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21,
                                                 domains=[(10, 50), (100, 150)])
        assert result is not None
        plt.close('all')
    
    def test_plot_peptide_custom_colors(self, mf):
        """Test plot with custom colors."""
        result = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21,
                                                 domain_color='blue',
                                                 domain_alpha=0.5)
        assert result is not None
        plt.close('all')
    
    def test_plot_peptide_save_to_file(self, mf, tmp_path):
        """Test saving plot to file."""
        fname = tmp_path / "test_peptide.png"
        result = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21,
                                                 fname=str(fname))
        assert fname.exists()
        plt.close('all')
    
    def test_plot_peptide_custom_ylim(self, mf):
        """Test plot with custom y limits."""
        result = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21,
                                                 ylim=[-2, 2])
        assert result is not None
        plt.close('all')
    
    def test_plot_peptide_with_grid(self, mf):
        """Test plot with grid enabled."""
        result = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21,
                                                 show_grid=True)
        assert result is not None
        plt.close('all')
    
    def test_plot_peptide_custom_vmin_vmax(self, mf):
        """Test plot with custom color scale limits."""
        result = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21,
                                                 vmin=-1.0, vmax=1.0)
        assert result is not None
        plt.close('all')
    
    def test_plot_peptide_custom_figsize(self, mf):
        """Test plot with custom figure size."""
        result = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21,
                                                 figsize=(8, 3))
        assert result is not None
        plt.close('all')
    
    def test_plot_peptide_different_peptides(self, mf):
        """Test plot with different peptides."""
        result1 = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21)
        plt.close('all')
        result2 = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_CHARGED, fragsize=21)
        plt.close('all')
        
        # Both should succeed
        assert result1 is not None
        assert result2 is not None
    
    def test_plot_peptide_zero_folded_false(self, mf):
        """Test plot with zero_folded disabled."""
        result = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21,
                                                 zero_folded=False)
        assert result is not None
        plt.close('all')
    
    def test_plot_peptide_no_smoothing(self, mf):
        """Test plot without smoothing."""
        result = mf.plot_protein_peptide_vector(LONG_SEQ_1, PEPTIDE_AROMATIC, fragsize=21,
                                                 smoothing_window=False)
        assert result is not None
        plt.close('all')


# =============================================================================
# BUILD PHASE DIAGRAM TESTS
# =============================================================================

class TestBuildPhaseDiagram:
    """Tests for the build_phase_diagram() function."""
    
    def test_phase_diagram_returns_list(self, mf):
        """Test that build_phase_diagram returns a list."""
        result = mf.build_phase_diagram(IDP_SEQ)
        assert isinstance(result, list)
    
    def test_phase_diagram_structure(self, mf):
        """Test the structure of the phase diagram list."""
        result = mf.build_phase_diagram(IDP_SEQ)
        
        # Should have 8 elements according to docstring
        assert len(result) == 8
        
        # Unpack and check types
        dilute, dense, critical, temps, dilute_spin, dense_spin, critical_spin, temps_spin = result
        
        # These are lists, not arrays
        assert isinstance(dilute, list)
        assert isinstance(dense, list)
        assert isinstance(critical, list)
    
    def test_phase_diagram_different_sequences(self, mf):
        """Test phase diagram gives different results for different sequences."""
        result_idp = mf.build_phase_diagram(IDP_SEQ)
        result_charged = mf.build_phase_diagram(POSITIVE_SEQ)
        
        # Different sequences should give different critical temperatures
        assert result_idp[2][1] != result_charged[2][1]


# =============================================================================
# PLOT PHASE DIAGRAM TESTS
# =============================================================================

class TestPlotPhaseDiagram:
    """Tests for the plot_phase_diagram() function."""
    
    def test_plot_phase_diagram_returns_list(self, mf):
        """Test that plot_phase_diagram returns a list."""
        result = mf.plot_phase_diagram(IDP_SEQ)
        assert isinstance(result, list)
        assert len(result) == 3
        
        B, fig, ax = result
        assert B is not None
        assert fig is not None
        assert ax is not None
        plt.close('all')
    
    def test_plot_phase_diagram_custom_style(self, mf):
        """Test plot with custom line style."""
        result = mf.plot_phase_diagram(IDP_SEQ, 
                                        line_color='blue',
                                        line_style='--',
                                        line_width=1.0)
        assert result is not None
        plt.close('all')
    
    def test_plot_phase_diagram_custom_limits(self, mf):
        """Test plot with custom axis limits."""
        result = mf.plot_phase_diagram(IDP_SEQ, 
                                        xlim=[0, 0.5],
                                        ylim=[0, 2])
        assert result is not None
        plt.close('all')
    
    def test_plot_phase_diagram_log_scale(self, mf):
        """Test plot with log x scale."""
        result = mf.plot_phase_diagram(IDP_SEQ, xlog=True)
        assert result is not None
        plt.close('all')
    
    def test_plot_phase_diagram_save_to_file(self, mf, tmp_path):
        """Test saving plot to file."""
        fname = tmp_path / "test_phase_diagram.png"
        result = mf.plot_phase_diagram(IDP_SEQ, filename=str(fname))
        assert fname.exists()
        plt.close('all')
    
    def test_plot_phase_diagram_custom_size(self, mf):
        """Test plot with custom figure size."""
        result = mf.plot_phase_diagram(IDP_SEQ, width=4, height=3)
        assert result is not None
        plt.close('all')


# =============================================================================
# PLOT MULTIPLE PHASE DIAGRAMS TESTS
# =============================================================================

class TestPlotMultiplePhaseDiagrams:
    """Tests for the plot_multiple_phase_diagrams() function."""
    
    def test_plot_multiple_returns_list(self, mf):
        """Test that plot_multiple_phase_diagrams returns a list."""
        seq_dict = {
            'seq1': [IDP_SEQ, 'blue'],
            'seq2': [MEDIUM_SEQ_1, 'red'],
        }
        result = mf.plot_multiple_phase_diagrams(seq_dict)
        assert isinstance(result, list)
        assert len(result) == 3
        
        all_B, fig, ax = result
        assert all_B is not None
        assert fig is not None
        assert ax is not None
        plt.close('all')
    
    def test_plot_multiple_with_tc_ref(self, mf):
        """Test plot with temperature normalization reference."""
        seq_dict = {
            'seq1': [IDP_SEQ, 'blue'],
            'seq2': [MEDIUM_SEQ_1, 'red'],
        }
        result = mf.plot_multiple_phase_diagrams(seq_dict, tc_ref='seq1')
        assert result is not None
        plt.close('all')
    
    def test_plot_multiple_tc_ref_not_in_dict(self, mf):
        """Test that invalid tc_ref raises error."""
        seq_dict = {
            'seq1': [IDP_SEQ, 'blue'],
        }
        with pytest.raises(ValueError):
            mf.plot_multiple_phase_diagrams(seq_dict, tc_ref='nonexistent')
        plt.close('all')
    
    def test_plot_multiple_save_to_file(self, mf, tmp_path):
        """Test saving multiple phase diagrams to file."""
        seq_dict = {
            'seq1': [IDP_SEQ, 'blue'],
            'seq2': [MEDIUM_SEQ_1, 'red'],
        }
        fname = tmp_path / "test_multiple_phase.png"
        result = mf.plot_multiple_phase_diagrams(seq_dict, filename=str(fname))
        assert fname.exists()
        plt.close('all')
    
    def test_plot_multiple_custom_style(self, mf):
        """Test multiple plots with custom style."""
        seq_dict = {
            'seq1': [IDP_SEQ, 'blue'],
            'seq2': [MEDIUM_SEQ_1, 'red'],
        }
        result = mf.plot_multiple_phase_diagrams(seq_dict,
                                                  line_style='--',
                                                  line_width=1.5,
                                                  xlog=True)
        assert result is not None
        plt.close('all')


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_epsilon_matches_matrix_center(self, mf):
        """Test that epsilon roughly corresponds to matrix average."""
        eps = mf.epsilon(LONG_SEQ_1, LONG_SEQ_2)
        B, _, _ = mf.intermolecular_idr_matrix(LONG_SEQ_1, LONG_SEQ_2, window_size=31)
        
        # The epsilon and matrix average won't be exactly equal due to
        # window effects, but they should be correlated
        matrix_mean = np.mean(B[0])
        
        # Both should have the same sign at least
        assert (eps < 0) == (matrix_mean < 0) or abs(eps) < 0.5 or abs(matrix_mean) < 0.5
    
    def test_custom_salt_affects_results(self, mf, mf_custom_salt):
        """Test that custom salt concentration affects results."""
        eps_default = mf.epsilon(POSITIVE_SEQ, NEGATIVE_SEQ)
        eps_custom = mf_custom_salt.epsilon(POSITIVE_SEQ, NEGATIVE_SEQ)
        
        # Different salt should give different epsilon for charged sequences
        assert not np.isclose(eps_default, eps_custom, rtol=0.01)
    
    def test_rna_sequence_handling(self, mf):
        """Test that RNA sequences (with U) are handled correctly."""
        # This tests the Mpipi_frontend's handling of U in sequences
        result = mf.intermolecular_idr_matrix(LONG_SEQ_1, RNA_SEQ, window_size=21)
        
        # Disorder should be disabled for RNA sequence
        disorder_2 = result[2]
        assert np.all(disorder_2 == 1)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_very_short_sequence(self, mf):
        """Test with very short sequences."""
        short = "AEKLS"
        result = mf.epsilon(short, short)
        assert isinstance(result, (int, float, np.floating))
    
    def test_single_residue_type(self, mf):
        """Test with single residue type sequences."""
        seq_a = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        seq_k = "KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK"
        
        eps_aa = mf.epsilon(seq_a, seq_a)
        eps_kk = mf.epsilon(seq_k, seq_k)
        eps_ak = mf.epsilon(seq_a, seq_k)
        
        assert all(isinstance(x, (int, float, np.floating)) for x in [eps_aa, eps_kk, eps_ak])
    
    def test_identical_sequences(self, mf):
        """Test that identical sequences give consistent results."""
        eps1 = mf.epsilon(MEDIUM_SEQ_1, MEDIUM_SEQ_1)
        eps2 = mf.epsilon(MEDIUM_SEQ_1, MEDIUM_SEQ_1)
        
        assert np.isclose(eps1, eps2, rtol=1e-10)


# =============================================================================
# DMS (DEEP MUTATIONAL SCANNING) TESTS
# =============================================================================

class TestDMS:
    """Tests for the dms() deep mutational scanning function."""
    
    # Short sequence for DMS testing (faster)
    DMS_TEST_SEQ = "AEKLSQPGWY"  # 10 aa
    
    def test_dms_returns_tuple(self, mf):
        """Test that dms returns a tuple with 3 elements."""
        result = mf.dms(self.DMS_TEST_SEQ, show_progress=False)
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_dms_matrix_shape(self, mf):
        """Test that the DMS matrix has correct shape (20 x n)."""
        seq = self.DMS_TEST_SEQ
        matrix, aas, positions = mf.dms(seq, show_progress=False)
        
        assert matrix.shape == (20, len(seq))
    
    def test_dms_amino_acids_list(self, mf):
        """Test that amino acids list is correct."""
        matrix, aas, positions = mf.dms(self.DMS_TEST_SEQ, show_progress=False)
        
        expected_aas = ['W', 'Y', 'F', 'I', 'L', 'V', 'M', 'A', 'G', 'S',                             
                             'T', 'N', 'Q', 'H', 'R', 'K', 'P', 'D', 'E', 'C']
        assert aas == expected_aas
        assert len(aas) == 20
    
    def test_dms_positions_array(self, mf):
        """Test that positions array is correct (1-indexed)."""
        seq = self.DMS_TEST_SEQ
        matrix, aas, positions = mf.dms(seq, show_progress=False)
        
        assert isinstance(positions, np.ndarray)
        assert len(positions) == len(seq)
        assert positions[0] == 1  # 1-indexed
        assert positions[-1] == len(seq)
    
    def test_dms_diagonal_matches_epsilon(self, mf):
        """Test that wild-type positions match direct epsilon calculation."""
        seq = "AEKLS"  # Very short for speed
        matrix, aas, positions = mf.dms(seq, show_progress=False)
        
        # Calculate epsilon for original sequence
        wt_epsilon = mf.epsilon(seq, seq)
        
        # For each position, the wild-type amino acid should give the same epsilon
        for pos_idx, aa in enumerate(seq):
            aa_idx = aas.index(aa)
            assert np.isclose(matrix[aa_idx, pos_idx], wt_epsilon, rtol=1e-10), \
                f"Mismatch at position {pos_idx+1} ({aa}): matrix={matrix[aa_idx, pos_idx]}, wt={wt_epsilon}"
    
    def test_dms_mutation_changes_epsilon(self, mf):
        """Test that mutations generally change the epsilon value."""
        seq = "KKKKK"  # All lysines
        matrix, aas, positions = mf.dms(seq, show_progress=False)
        
        # Mutating to glutamate (opposite charge) should change epsilon significantly
        k_idx = aas.index('K')
        e_idx = aas.index('E')
        
        wt_epsilon = matrix[k_idx, 0]  # Wild-type (K at position 1)
        mutant_epsilon = matrix[e_idx, 0]  # K->E at position 1
        
        assert wt_epsilon != mutant_epsilon
    
    def test_dms_symmetric_sequence(self, mf):
        """Test DMS on a symmetric sequence."""
        seq = "AEKKE"  # Palindromic-ish
        matrix, aas, positions = mf.dms(seq, show_progress=False)
        
        # Matrix should have valid values
        assert np.all(np.isfinite(matrix))
    
    def test_dms_weighting_options(self, mf):
        """Test DMS with different weighting options."""
        seq = "AEKLM"  # Short seq with aliphatic
        
        matrix_default, _, _ = mf.dms(seq, show_progress=False)
        matrix_no_aliphatic, _, _ = mf.dms(seq, 
                                            use_aliphatic_weighting=False,
                                            show_progress=False)
        matrix_no_charge, _, _ = mf.dms(seq, 
                                         use_charge_weighting=False,
                                         show_progress=False)
        
        # Results should differ with different weightings
        assert not np.allclose(matrix_default, matrix_no_aliphatic)
        assert not np.allclose(matrix_default, matrix_no_charge)
    
    def test_dms_all_same_residue(self, mf):
        """Test DMS on a sequence of all same residues."""
        seq = "AAAAA"
        matrix, aas, positions = mf.dms(seq, show_progress=False)
        
        # All positions should have the same values for each amino acid
        # since the sequence is uniform
        for aa_idx in range(20):
            row = matrix[aa_idx, :]
            # All values in each row should be the same
            assert np.allclose(row, row[0]), \
                f"Row {aas[aa_idx]} not uniform: {row}"
    
    def test_dms_matrix_values_are_floats(self, mf):
        """Test that all matrix values are valid floats."""
        matrix, aas, positions = mf.dms(self.DMS_TEST_SEQ, show_progress=False)
        
        assert matrix.dtype in [np.float64, np.float32, float]
        assert np.all(np.isfinite(matrix))
    
    def test_dms_charged_mutations(self, mf):
        """Test that charge-changing mutations have expected effects."""
        seq = "EEEEE"  # All negative
        matrix, aas, positions = mf.dms(seq, show_progress=False)
        
        e_idx = aas.index('E')
        k_idx = aas.index('K')
        
        # Wild-type epsilon (all E)
        wt_eps = matrix[e_idx, 0]
        
        # Mutating E->K should make it less self-attractive (less negative/more positive)
        # because we're introducing opposite charges
        mutant_eps = matrix[k_idx, 0]
        
        # The change in epsilon depends on the model, but they should be different
        assert wt_eps != mutant_eps

    def test_dms_return_delta(self, mf):
        """Test DMS with return_delta=True returns differences from WT."""
        seq = "AEKLS"
        
        # Get absolute values
        matrix_abs, aas, positions = mf.dms(seq, return_delta=False, show_progress=False)
        
        # Get delta values
        matrix_delta, _, _ = mf.dms(seq, return_delta=True, show_progress=False)
        
        # Calculate WT epsilon
        wt_epsilon = mf.epsilon(seq, seq)
        
        # Delta should be absolute - WT
        expected_delta = matrix_abs - wt_epsilon
        assert np.allclose(matrix_delta, expected_delta, rtol=1e-10)
    
    def test_dms_return_delta_wt_is_zero(self, mf):
        """Test that wild-type positions have delta=0 when return_delta=True."""
        seq = "AEKLS"
        matrix_delta, aas, positions = mf.dms(seq, return_delta=True, show_progress=False)
        
        # For each position, the WT amino acid should have delta=0
        for pos_idx, wt_aa in enumerate(seq):
            aa_idx = aas.index(wt_aa)
            assert np.isclose(matrix_delta[aa_idx, pos_idx], 0, atol=1e-10), \
                f"WT at position {pos_idx+1} ({wt_aa}) has non-zero delta: {matrix_delta[aa_idx, pos_idx]}"


# =============================================================================
# PLOT DMS TESTS
# =============================================================================

class TestPlotDMS:
    """Tests for the plot_dms() function."""
    
    DMS_PLOT_SEQ = "AEKLSQPGWY"  # 10 aa for faster tests
    
    def test_plot_dms_returns_tuple(self, mf):
        """Test that plot_dms returns a tuple with 4 elements."""
        result = mf.plot_dms(self.DMS_PLOT_SEQ, show_progress=False)
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        fig, ax, im, dms_data = result
        assert fig is not None
        assert ax is not None
        assert im is not None
        assert isinstance(dms_data, tuple)
        plt.close('all')
    
    def test_plot_dms_dms_data_structure(self, mf):
        """Test that the returned DMS data has correct structure."""
        fig, ax, im, dms_data = mf.plot_dms(self.DMS_PLOT_SEQ, show_progress=False)
        
        matrix, aas, positions = dms_data
        assert matrix.shape == (20, len(self.DMS_PLOT_SEQ))
        assert len(aas) == 20
        assert len(positions) == len(self.DMS_PLOT_SEQ)
        plt.close('all')
    
    def test_plot_dms_return_delta_true(self, mf):
        """Test plot_dms with return_delta=True (default)."""
        fig, ax, im, dms_data = mf.plot_dms(self.DMS_PLOT_SEQ, 
                                             return_delta=True,
                                             show_progress=False)
        matrix = dms_data[0]
        aas = dms_data[1]
        
        # WT positions should have delta=0
        for pos_idx, wt_aa in enumerate(self.DMS_PLOT_SEQ):
            aa_idx = aas.index(wt_aa)
            assert np.isclose(matrix[aa_idx, pos_idx], 0, atol=1e-10)
        plt.close('all')
    
    def test_plot_dms_return_delta_false(self, mf):
        """Test plot_dms with return_delta=False."""
        fig, ax, im, dms_data = mf.plot_dms(self.DMS_PLOT_SEQ, 
                                             return_delta=False,
                                             show_progress=False)
        matrix = dms_data[0]
        
        # All values should be absolute (typically negative for self-attraction)
        assert np.all(np.isfinite(matrix))
        plt.close('all')
    
    def test_plot_dms_custom_colormap(self, mf):
        """Test plot_dms with custom colormap."""
        fig, ax, im, _ = mf.plot_dms(self.DMS_PLOT_SEQ, 
                                      cmap='coolwarm',
                                      show_progress=False)
        assert fig is not None
        plt.close('all')
    
    def test_plot_dms_custom_vmin_vmax(self, mf):
        """Test plot_dms with custom color scale limits."""
        fig, ax, im, _ = mf.plot_dms(self.DMS_PLOT_SEQ, 
                                      vmin=-5, vmax=5,
                                      show_progress=False)
        assert fig is not None
        plt.close('all')
    
    def test_plot_dms_custom_figsize(self, mf):
        """Test plot_dms with custom figure size."""
        fig, ax, im, _ = mf.plot_dms(self.DMS_PLOT_SEQ, 
                                      figsize=(10, 8),
                                      show_progress=False)
        assert fig is not None
        plt.close('all')
    
    def test_plot_dms_no_wt_marker(self, mf):
        """Test plot_dms without wild-type markers."""
        fig, ax, im, _ = mf.plot_dms(self.DMS_PLOT_SEQ, 
                                      show_wt_marker=False,
                                      show_progress=False)
        assert fig is not None
        plt.close('all')
    
    def test_plot_dms_custom_wt_marker(self, mf):
        """Test plot_dms with custom wild-type marker style."""
        fig, ax, im, _ = mf.plot_dms(self.DMS_PLOT_SEQ, 
                                      wt_marker='x',
                                      wt_marker_color='red',
                                      wt_marker_size=5,
                                      show_progress=False)
        assert fig is not None
        plt.close('all')
    
    def test_plot_dms_save_to_file(self, mf, tmp_path):
        """Test saving plot_dms to file."""
        fname = tmp_path / "test_dms_heatmap.png"
        fig, ax, im, _ = mf.plot_dms(self.DMS_PLOT_SEQ, 
                                      fname=str(fname),
                                      show_progress=False)
        assert fname.exists()
        plt.close('all')
    
    def test_plot_dms_tic_frequency(self, mf):
        """Test plot_dms with custom tick frequency."""
        fig, ax, im, _ = mf.plot_dms(self.DMS_PLOT_SEQ, 
                                      tic_frequency=5,
                                      show_progress=False)
        assert fig is not None
        plt.close('all')
    
    def test_plot_dms_longer_sequence(self, mf):
        """Test plot_dms with a longer sequence."""
        long_seq = "MSKGEELFTGVVPILVELDGDVNGHK"  # 26 aa
        fig, ax, im, dms_data = mf.plot_dms(long_seq, show_progress=False)
        
        assert dms_data[0].shape == (20, len(long_seq))
        plt.close('all')
    
    def test_plot_dms_show_sequence(self, mf):
        """Test plot_dms with sequence display enabled."""
        fig, ax, im, _ = mf.plot_dms(self.DMS_PLOT_SEQ, 
                                      show_sequence=True,
                                      show_progress=False)
        assert fig is not None
        plt.close('all')
    
    def test_plot_dms_show_sequence_custom_fontsize(self, mf):
        """Test plot_dms with sequence display and custom font size."""
        fig, ax, im, _ = mf.plot_dms(self.DMS_PLOT_SEQ, 
                                      show_sequence=True,
                                      sequence_fontsize=8,
                                      show_progress=False)
        assert fig is not None
        plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
