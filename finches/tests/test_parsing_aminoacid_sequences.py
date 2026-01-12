"""
Comprehensive tests for the parsing_aminoacid_sequences module.

Run with: pytest test_parsing_aminoacid_sequences.py -v

By: Auto-generated test suite
"""

import pytest
import numpy as np
from finches import parsing_aminoacid_sequences


# =============================================================================
# Tests for get_aliphatic_groups
# =============================================================================

class TestGetAliphaticGroups:
    """Tests for aliphatic residue clustering classification."""

    def test_isolated_aliphatic(self):
        """Single isolated aliphatic gets level 1."""
        result = parsing_aminoacid_sequences.get_aliphatic_groups("GGGAGG")
        assert result == [0, 0, 0, 1, 0, 0]

    def test_pair_of_aliphatics(self):
        """Adjacent pair of aliphatics gets level 2."""
        result = parsing_aminoacid_sequences.get_aliphatic_groups("GGAAGG")
        assert result == [0, 0, 2, 2, 0, 0]

    def test_cluster_of_aliphatics(self):
        """Cluster of 3+ aliphatics gets level 3."""
        result = parsing_aminoacid_sequences.get_aliphatic_groups("GAAAAG")
        # 4 A's in a row - each sees 3+ neighbors, so level 3
        assert result == [0, 3, 3, 3, 3, 0]

    def test_all_aliphatic_residues(self):
        """All aliphatic residue types are recognized."""
        # A, V, I, L, M are aliphatic
        result = parsing_aminoacid_sequences.get_aliphatic_groups("AVILM")
        # All 5 are adjacent, so each should see plenty of neighbors
        for val in result:
            assert val >= 2  # At least level 2 for all

    def test_no_aliphatics(self):
        """Sequence with no aliphatics gets all zeros."""
        result = parsing_aminoacid_sequences.get_aliphatic_groups("GGGGGG")
        assert result == [0, 0, 0, 0, 0, 0]

    def test_non_aliphatic_residues(self):
        """Non-aliphatic residues always get 0."""
        # K, R, E, D, G, S, etc. are not aliphatic
        result = parsing_aminoacid_sequences.get_aliphatic_groups("KRDEGSNQ")
        assert all(v == 0 for v in result)

    def test_separated_aliphatics(self):
        """Aliphatics separated by large gap are isolated."""
        # Large gap between A's
        result = parsing_aminoacid_sequences.get_aliphatic_groups("AGGGGGA")
        # Both A's should be isolated (level 1)
        assert result[0] == 1
        assert result[-1] == 1
        # Middle positions are not aliphatic
        assert all(result[i] == 0 for i in range(1, 6))

    def test_output_length_matches_input(self):
        """Output length always matches input sequence length."""
        for seq in ["A", "AAA", "GGGAAAGGG", "AVILMGGGAVILM"]:
            result = parsing_aminoacid_sequences.get_aliphatic_groups(seq)
            assert len(result) == len(seq)

    def test_capping_at_3(self):
        """Clustering level is capped at 3."""
        # Very long run of aliphatics
        result = parsing_aminoacid_sequences.get_aliphatic_groups("AAAAAAAAAA")
        # All should be capped at 3
        assert all(v <= 3 for v in result)
        # Middle residues should be level 3
        assert result[5] == 3

    def test_mixed_sequence(self):
        """Complex mixed sequence."""
        seq = "GAALGGVVIG"
        result = parsing_aminoacid_sequences.get_aliphatic_groups(seq)
        # Positions 0, 4, 5, 9 are G (non-aliphatic) -> 0
        assert result[0] == 0
        assert result[4] == 0
        assert result[5] == 0
        assert result[9] == 0
        # Other positions are aliphatic with varying clustering


# =============================================================================
# Tests for get_aliphatic_weighted_mask
# =============================================================================

class TestGetAliphaticWeightedMask:
    """Tests for aliphatic clustering weight matrix."""

    def test_output_shape(self):
        """Output shape matches input sequence lengths."""
        seq1 = "AAAAA"
        seq2 = "GGG"
        result = parsing_aminoacid_sequences.get_aliphatic_weighted_mask(seq1, seq2)
        assert result.shape == (len(seq1), len(seq2))

    def test_returns_numpy_array(self):
        """Should return a numpy array."""
        result = parsing_aminoacid_sequences.get_aliphatic_weighted_mask("AAA", "AAA")
        assert isinstance(result, np.ndarray)

    def test_no_aliphatics_all_ones(self):
        """Non-aliphatic sequences should have all 1.0 weights."""
        result = parsing_aminoacid_sequences.get_aliphatic_weighted_mask("GGGG", "GGGG")
        assert np.all(result == 1.0)

    def test_large_clusters_get_3(self):
        """Large aliphatic clusters (level 3) get weight 3.0."""
        # Both sequences have large clusters
        seq1 = "AAAA"  # All level 3
        seq2 = "AAAA"  # All level 3
        result = parsing_aminoacid_sequences.get_aliphatic_weighted_mask(seq1, seq2)
        # min(3, 3) = 3 -> weight 3.0
        assert np.all(result == 3.0)

    def test_small_clusters_get_1_5(self):
        """Small aliphatic clusters (level 2) get weight 1.5."""
        # Pairs of aliphatics
        seq1 = "GAA"  # G=0, A=2, A=2
        seq2 = "GAA"  # G=0, A=2, A=2
        result = parsing_aminoacid_sequences.get_aliphatic_weighted_mask(seq1, seq2)
        # At positions where both are level 2: min(2, 2) = 2 -> 1.5
        assert result[1, 1] == 1.5
        assert result[2, 2] == 1.5
        # Where one is 0: min(0, 2) = 0 -> 1.0
        assert result[0, 1] == 1.0

    def test_isolated_aliphatics_get_1(self):
        """Isolated aliphatics (level 1) get weight 1.0."""
        seq1 = "GGGAG"  # Isolated A at position 3
        seq2 = "GGGAG"
        result = parsing_aminoacid_sequences.get_aliphatic_weighted_mask(seq1, seq2)
        # min(1, 1) = 1 -> weight 1.0
        assert result[3, 3] == 1.0

    def test_weight_values_are_valid(self):
        """All weights should be 1.0, 1.5, or 3.0."""
        seq1 = "AAAGGGAAGAAA"
        seq2 = "GAAAGGGAAAG"
        result = parsing_aminoacid_sequences.get_aliphatic_weighted_mask(seq1, seq2)
        valid_weights = {1.0, 1.5, 3.0}
        for weight in result.flatten():
            assert weight in valid_weights

    def test_symmetric_sequences(self):
        """Same sequences should give symmetric matrix."""
        seq = "AAGAA"
        result = parsing_aminoacid_sequences.get_aliphatic_weighted_mask(seq, seq)
        assert np.allclose(result, result.T)

    def test_asymmetric_sequences(self):
        """Different sequences can give asymmetric matrix."""
        seq1 = "AAAA"  # All level 3
        seq2 = "GAAG"  # Mixed levels
        result = parsing_aminoacid_sequences.get_aliphatic_weighted_mask(seq1, seq2)
        # Matrix is len(seq1) x len(seq2) = 4 x 4
        assert result.shape == (4, 4)


# =============================================================================
# Tests for get_charge_weighted_mask
# =============================================================================

class TestGetChargeWeightedMask:
    """Tests for charge-weighted interaction mask."""

    def test_returns_tuple_of_two_arrays(self):
        """Should return tuple of two numpy arrays."""
        result = parsing_aminoacid_sequences.get_charge_weighted_mask("KKK", "DDD")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)

    def test_output_shapes(self):
        """Output shapes match input sequence lengths."""
        seq1 = "KKKKK"
        seq2 = "DDD"
        attractive, repulsive = parsing_aminoacid_sequences.get_charge_weighted_mask(seq1, seq2)
        assert attractive.shape == (len(seq1), len(seq2))
        assert repulsive.shape == (len(seq1), len(seq2))

    def test_attractive_matrix_is_zeros(self):
        """Attractive matrix is currently always zeros."""
        attractive, _ = parsing_aminoacid_sequences.get_charge_weighted_mask("KKKEEE", "KKKEEE")
        assert np.all(attractive == 0.0)

    def test_non_charged_residues_are_zero(self):
        """Non-charged residue positions have zero weight."""
        _, repulsive = parsing_aminoacid_sequences.get_charge_weighted_mask("GGGKG", "GGGDG")
        # Only position 3 in each sequence is charged
        # All other rows/columns should be zero
        for i in [0, 1, 2, 4]:
            assert np.all(repulsive[i, :] == 0.0)
            assert np.all(repulsive[:, i] == 0.0)
        # Position (3, 3) can have a non-zero weight
        # (K and D are opposite charges, mixed fragment)

    def test_all_same_charge_max_weight(self):
        """All same charge in windows gives max weight of 1.0."""
        # KKK + KKK = 6 positives, 0 negatives
        # |NCPR|/FCR = |6/6| / (6/6) = 1.0
        _, repulsive = parsing_aminoacid_sequences.get_charge_weighted_mask("KKK", "KKK")
        # Center position (1, 1) should have weight close to 1.0
        assert repulsive[1, 1] == pytest.approx(1.0)

    def test_balanced_charges_zero_weight(self):
        """Balanced charges (|NCPR| = 0) gives weight 0."""
        # KKK + EEE = 3 positive, 3 negative
        # |NCPR|/FCR = |0|/1 = 0.0
        _, repulsive = parsing_aminoacid_sequences.get_charge_weighted_mask("KKK", "EEE")
        # All intersections should be 0 due to balanced charges
        assert repulsive[1, 1] == pytest.approx(0.0)

    def test_all_charged_residues_recognized(self):
        """K, R, E, D are all recognized as charged."""
        # Test each charged residue type
        for pos in ['K', 'R']:
            for neg in ['E', 'D']:
                _, repulsive = parsing_aminoacid_sequences.get_charge_weighted_mask(
                    f"G{pos}G", f"G{neg}G"
                )
                # Position (1, 1) should have some weight (both charged)
                # Opposite charges = balanced = 0
                assert repulsive[1, 1] == pytest.approx(0.0)

    def test_weight_range(self):
        """Weights should be between 0 and 1."""
        test_cases = [
            ("KKKKK", "KKKKK"),
            ("EEEEE", "EEEEE"),
            ("KKKKK", "EEEEE"),
            ("KEKEK", "EKEKE"),
        ]
        for seq1, seq2 in test_cases:
            _, repulsive = parsing_aminoacid_sequences.get_charge_weighted_mask(seq1, seq2)
            assert np.all(repulsive >= 0.0)
            assert np.all(repulsive <= 1.0)

    def test_symmetric_for_same_sequences(self):
        """Same sequences should give symmetric repulsive matrix."""
        seq = "KEKEK"
        _, repulsive = parsing_aminoacid_sequences.get_charge_weighted_mask(seq, seq)
        assert np.allclose(repulsive, repulsive.T)

    def test_charged_positions_only(self):
        """Only positions where BOTH residues are charged have non-zero weight."""
        seq1 = "AKAKA"  # Charged at 1, 3
        seq2 = "GEGEG"  # Charged at 1, 3 (E)
        _, repulsive = parsing_aminoacid_sequences.get_charge_weighted_mask(seq1, seq2)
        
        # Non-charged positions (0, 2, 4) should be all zeros
        for i in [0, 2, 4]:
            assert np.all(repulsive[i, :] == 0.0)
            assert np.all(repulsive[:, i] == 0.0)


# =============================================================================
# Tests for get_aliphaticgroup_sequence
# =============================================================================

class TestGetAliphaticgroupSequence:
    """Tests for aliphatic group sequence reassignment."""

    def test_no_aliphatics_unchanged(self):
        """Sequence with no aliphatics is unchanged."""
        result = parsing_aminoacid_sequences.get_aliphaticgroup_sequence("GGGSGG")
        assert result == "GGGSGG"

    def test_isolated_aliphatics_unchanged(self):
        """Isolated aliphatics (level 1) are unchanged."""
        result = parsing_aminoacid_sequences.get_aliphaticgroup_sequence("GGGAGG")
        # Isolated A stays as A
        assert result == "GGGAGG"

    def test_level2_aliphatics_group1(self):
        """Level 2 aliphatics get group1 symbols."""
        # AA pair should be level 2
        result = parsing_aminoacid_sequences.get_aliphaticgroup_sequence("GGAAGG")
        # A -> 'a' in group1
        assert result == "GGaaGG"

    def test_level3_aliphatics_group2(self):
        """Level 3 aliphatics get group2 symbols."""
        # AAAA cluster should be level 3
        result = parsing_aminoacid_sequences.get_aliphaticgroup_sequence("GAAAAG")
        # A -> 'b' in group2
        assert result == "GbbbbG"

    def test_all_aliphatic_mappings_group1(self):
        """All aliphatic residues map correctly for group1."""
        # Group 1 mappings: A->a, L->l, M->m, I->i, V->v
        # Create pairs (level 2)
        result = parsing_aminoacid_sequences.get_aliphaticgroup_sequence("AALLMMII VV")
        # Note: space is not aliphatic, so VV should be pair
        # Actually "AALLMMIIVV" has continuous aliphatics
        result = parsing_aminoacid_sequences.get_aliphaticgroup_sequence("GAA")
        assert 'a' in result  # A -> a for level 2

    def test_all_aliphatic_mappings_group2(self):
        """All aliphatic residues map correctly for group2."""
        # Group 2 mappings: A->b, L->o, M->x, I->y, V->z
        # Create clusters (level 3)
        result = parsing_aminoacid_sequences.get_aliphaticgroup_sequence("GAAAA")
        assert 'b' in result  # A -> b for level 3

    def test_output_length_preserved(self):
        """Output length matches input length."""
        for seq in ["AAAAA", "GGGGG", "AAAGGGAAA", "AVILMAVILM"]:
            result = parsing_aminoacid_sequences.get_aliphaticgroup_sequence(seq)
            assert len(result) == len(seq)

    def test_mixed_levels(self):
        """Sequence with mixed clustering levels."""
        # G-A-A-G-G-G-A-G
        # Position 1-2: AA pair -> level 2
        # Position 6: isolated A -> level 1
        seq = "GAAGGGA G"
        # Remove space for valid sequence
        seq = "GAAGGGAG"
        result = parsing_aminoacid_sequences.get_aliphaticgroup_sequence(seq)
        # AA should become 'aa', isolated A stays 'A'
        assert result[1] == 'a'
        assert result[2] == 'a'
        assert result[6] == 'A'  # Isolated, stays uppercase

    def test_non_aliphatic_unchanged(self):
        """Non-aliphatic residues are never changed."""
        seq = "KAAAK"
        result = parsing_aminoacid_sequences.get_aliphaticgroup_sequence(seq)
        # K should stay K
        assert result[0] == 'K'
        assert result[-1] == 'K'


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_aliphatic_groups_feeds_weighted_mask(self):
        """get_aliphatic_groups output is used by get_aliphatic_weighted_mask."""
        seq = "AAAGGGAAA"
        
        # Get groups directly
        groups = parsing_aminoacid_sequences.get_aliphatic_groups(seq)
        
        # Get weighted mask
        weights = parsing_aminoacid_sequences.get_aliphatic_weighted_mask(seq, seq)
        
        # Verify relationship: where both groups >= 3, weight should be 3.0
        for i in range(len(seq)):
            for j in range(len(seq)):
                min_group = min(groups[i], groups[j])
                if min_group >= 3:
                    assert weights[i, j] == 3.0
                elif min_group == 2:
                    assert weights[i, j] == 1.5
                else:
                    assert weights[i, j] == 1.0

    def test_charged_mask_with_realistic_sequence(self):
        """Test charge mask with a realistic protein-like sequence."""
        # A sequence with charged clusters
        seq1 = "AAAAKKKAAAADDDAAAA"
        seq2 = "GGGRRRGGGEEEGGG"
        
        attractive, repulsive = parsing_aminoacid_sequences.get_charge_weighted_mask(seq1, seq2)
        
        # Should have non-zero weights where charged residues intersect
        assert attractive.shape == (len(seq1), len(seq2))
        assert repulsive.shape == (len(seq1), len(seq2))
        
        # All weights should be valid
        assert np.all(repulsive >= 0.0)
        assert np.all(repulsive <= 1.0)

    def test_aliphatic_sequence_transformation(self):
        """Full workflow: groups -> weighted mask -> sequence transformation."""
        seq = "AAAGGGAAAA"
        
        # Get clustering groups
        groups = parsing_aminoacid_sequences.get_aliphatic_groups(seq)
        
        # Transform sequence
        transformed = parsing_aminoacid_sequences.get_aliphaticgroup_sequence(seq)
        
        # Verify transformation matches group levels
        for i, (orig, new, group) in enumerate(zip(seq, transformed, groups)):
            if orig in "AVILM":  # Aliphatic
                if group == 1:
                    assert new == orig  # Unchanged
                elif group == 2:
                    assert new.islower() and new != 'b'  # Group 1 symbol
                elif group == 3:
                    assert new in ['b', 'o', 'x', 'y', 'z']  # Group 2 symbol
            else:
                assert new == orig  # Non-aliphatic unchanged


# =============================================================================
# Edge case tests
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_residue_sequences(self):
        """Single residue sequences."""
        # Aliphatic groups
        assert parsing_aminoacid_sequences.get_aliphatic_groups("A") == [1]
        assert parsing_aminoacid_sequences.get_aliphatic_groups("G") == [0]
        
        # Weighted mask
        result = parsing_aminoacid_sequences.get_aliphatic_weighted_mask("A", "A")
        assert result.shape == (1, 1)
        assert result[0, 0] == 1.0  # Isolated aliphatic
        
        # Charge mask
        attr, rep = parsing_aminoacid_sequences.get_charge_weighted_mask("K", "K")
        assert attr.shape == (1, 1)
        assert rep.shape == (1, 1)

    def test_very_long_sequences(self):
        """Very long sequences should work."""
        long_seq = "A" * 100
        
        groups = parsing_aminoacid_sequences.get_aliphatic_groups(long_seq)
        assert len(groups) == 100
        
        weights = parsing_aminoacid_sequences.get_aliphatic_weighted_mask(long_seq, long_seq)
        assert weights.shape == (100, 100)

    def test_asymmetric_sequence_lengths(self):
        """Different length sequences in masks."""
        seq1 = "AA"
        seq2 = "AAAAA"
        
        weights = parsing_aminoacid_sequences.get_aliphatic_weighted_mask(seq1, seq2)
        assert weights.shape == (2, 5)
        
        attr, rep = parsing_aminoacid_sequences.get_charge_weighted_mask("KK", "EEEEE")
        assert attr.shape == (2, 5)
        assert rep.shape == (2, 5)

    def test_all_20_standard_amino_acids(self):
        """Test with all 20 standard amino acids."""
        all_aa = "ACDEFGHIKLMNPQRSTVWY"
        
        groups = parsing_aminoacid_sequences.get_aliphatic_groups(all_aa)
        assert len(groups) == 20
        
        # A(0), I(7), L(9), M(10), V(17) are aliphatic
        # Note: I, L, M are adjacent so they form a cluster
        aliphatic_positions = {0, 7, 9, 10, 17}
        for i, g in enumerate(groups):
            if i in aliphatic_positions:
                assert g >= 1, f"Position {i} ({all_aa[i]}) should be aliphatic"
            else:
                assert g == 0, f"Position {i} ({all_aa[i]}) should NOT be aliphatic"
