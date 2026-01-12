"""
Comprehensive tests for the sequence_tools module.

Run with: pytest test_sequence_tools.py -v

By: Auto-generated test suite
"""

import pytest
from finches import sequence_tools


# =============================================================================
# Tests for calculate_NCPR
# =============================================================================

class TestCalculateNCPR:
    """Tests for the net charge per residue (NCPR) calculation."""

    def test_all_positive(self):
        """Sequence with only positive residues."""
        # 5 positive residues (K, R)
        assert sequence_tools.calculate_NCPR("KKKKK") == 1.0
        assert sequence_tools.calculate_NCPR("RRRRR") == 1.0
        assert sequence_tools.calculate_NCPR("KRKRK") == 1.0

    def test_all_negative(self):
        """Sequence with only negative residues."""
        # 5 negative residues (D, E)
        assert sequence_tools.calculate_NCPR("DDDDD") == -1.0
        assert sequence_tools.calculate_NCPR("EEEEE") == -1.0
        assert sequence_tools.calculate_NCPR("DEDED") == -1.0

    def test_neutral_sequence(self):
        """Sequence with no charged residues."""
        assert sequence_tools.calculate_NCPR("AAAAA") == 0.0
        assert sequence_tools.calculate_NCPR("GGGGG") == 0.0

    def test_balanced_charges(self):
        """Sequence with equal positive and negative charges."""
        # 2 positive (K, K), 2 negative (D, D), 2 neutral (A, A)
        assert sequence_tools.calculate_NCPR("KKDDAA") == 0.0
        # 2 positive (K, R), 2 negative (D, E), 2 neutral (A, A)
        assert sequence_tools.calculate_NCPR("KRDAAE") == 0.0

    def test_mixed_charges(self):
        """Sequence with unequal positive and negative charges."""
        # 3 positive (K, K, R), 1 negative (D), total = +2, length = 5
        assert sequence_tools.calculate_NCPR("KKRAD") == pytest.approx(2/5)
        
        # 1 positive (K), 3 negative (D, D, E), total = -2, length = 5
        assert sequence_tools.calculate_NCPR("KADDE") == pytest.approx(-2/5)

    def test_single_residue(self):
        """Single residue sequences."""
        assert sequence_tools.calculate_NCPR("K") == 1.0
        assert sequence_tools.calculate_NCPR("D") == -1.0
        assert sequence_tools.calculate_NCPR("A") == 0.0


# =============================================================================
# Tests for calculate_FCR
# =============================================================================

class TestCalculateFCR:
    """Tests for the fraction of charged residues (FCR) calculation."""

    def test_all_charged(self):
        """Sequence with all charged residues."""
        assert sequence_tools.calculate_FCR("KKKKK") == 1.0
        assert sequence_tools.calculate_FCR("DDDDD") == 1.0
        assert sequence_tools.calculate_FCR("KDKDK") == 1.0
        assert sequence_tools.calculate_FCR("KRED") == 1.0

    def test_no_charged(self):
        """Sequence with no charged residues."""
        assert sequence_tools.calculate_FCR("AAAAA") == 0.0
        assert sequence_tools.calculate_FCR("GGGGG") == 0.0
        assert sequence_tools.calculate_FCR("AAGLM") == 0.0

    def test_mixed(self):
        """Sequence with some charged residues."""
        # 2 charged (K, D) out of 4
        assert sequence_tools.calculate_FCR("AKAD") == pytest.approx(0.5)
        
        # 3 charged (K, R, E) out of 6
        assert sequence_tools.calculate_FCR("AKRAGE") == pytest.approx(3/6)

    def test_single_residue(self):
        """Single residue sequences."""
        assert sequence_tools.calculate_FCR("K") == 1.0
        assert sequence_tools.calculate_FCR("A") == 0.0


# =============================================================================
# Tests for calculate_FCR_and_NCPR
# =============================================================================

class TestCalculateFCRandNCPR:
    """Tests for the combined FCR and NCPR calculation."""

    def test_returns_list(self):
        """Should return a list with two elements."""
        result = sequence_tools.calculate_FCR_and_NCPR("AKAD")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_fcr_ncpr_values(self):
        """Test correct FCR and NCPR values."""
        # 2 positive (K, K), 1 negative (D), 2 neutral (A, A)
        # FCR = 3/5, NCPR = 1/5
        result = sequence_tools.calculate_FCR_and_NCPR("AKKDA")
        assert result[0] == pytest.approx(3/5)  # FCR
        assert result[1] == pytest.approx(1/5)  # NCPR

    def test_all_positive(self):
        """All positive residues."""
        result = sequence_tools.calculate_FCR_and_NCPR("KKKKK")
        assert result[0] == 1.0  # FCR
        assert result[1] == 1.0  # NCPR

    def test_all_negative(self):
        """All negative residues."""
        result = sequence_tools.calculate_FCR_and_NCPR("DDDDD")
        assert result[0] == 1.0   # FCR
        assert result[1] == -1.0  # NCPR

    def test_balanced(self):
        """Equal positive and negative."""
        result = sequence_tools.calculate_FCR_and_NCPR("KKDD")
        assert result[0] == 1.0  # FCR (all charged)
        assert result[1] == 0.0  # NCPR (balanced)

    def test_neutral(self):
        """No charged residues."""
        result = sequence_tools.calculate_FCR_and_NCPR("AAAAA")
        assert result[0] == 0.0  # FCR
        assert result[1] == 0.0  # NCPR


# =============================================================================
# Tests for mask_sequence
# =============================================================================

class TestMaskSequence:
    """Tests for the sequence masking function."""

    def test_basic_masking(self):
        """Basic masking with list of targets."""
        result = sequence_tools.mask_sequence("ACDEFGK", ['A', 'G'])
        assert result == [1, 0, 0, 0, 0, 1, 0]

    def test_charged_residues(self):
        """Mask for charged residues."""
        result = sequence_tools.mask_sequence("KKEKK", ['K', 'R'])
        assert result == [1, 1, 0, 1, 1]

    def test_set_input(self):
        """Target residues as a set."""
        result = sequence_tools.mask_sequence("ACDEK", {'A', 'E'})
        assert result == [1, 0, 0, 1, 0]

    def test_no_matches(self):
        """No residues match the target."""
        result = sequence_tools.mask_sequence("AAAAA", ['K', 'R'])
        assert result == [0, 0, 0, 0, 0]

    def test_all_match(self):
        """All residues match the target."""
        result = sequence_tools.mask_sequence("KKKKK", ['K'])
        assert result == [1, 1, 1, 1, 1]

    def test_single_residue(self):
        """Single residue sequence."""
        assert sequence_tools.mask_sequence("K", ['K']) == [1]
        assert sequence_tools.mask_sequence("A", ['K']) == [0]

    def test_empty_targets(self):
        """Empty target list."""
        result = sequence_tools.mask_sequence("ACDEF", [])
        assert result == [0, 0, 0, 0, 0]

    def test_length_matches(self):
        """Output length matches input length."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        result = sequence_tools.mask_sequence(seq, ['A', 'G', 'P'])
        assert len(result) == len(seq)


# =============================================================================
# Tests for get_neighbors_window_of3
# =============================================================================

class TestGetNeighborsWindowOf3:
    """Tests for the 3-residue window extraction."""

    def test_middle_position(self):
        """Middle of sequence - full window."""
        result = sequence_tools.get_neighbors_window_of3(5, "ACDEFGHIK")
        assert result == "FGH"

    def test_start_position(self):
        """Start of sequence - truncated window."""
        result = sequence_tools.get_neighbors_window_of3(0, "ACDEFGHIK")
        assert result == "AC"

    def test_end_position(self):
        """End of sequence - truncated window."""
        result = sequence_tools.get_neighbors_window_of3(8, "ACDEFGHIK")
        assert result == "IK"

    def test_position_one(self):
        """Second position - full window."""
        result = sequence_tools.get_neighbors_window_of3(1, "ACDEFGHIK")
        assert result == "ACD"

    def test_second_to_last(self):
        """Second to last position - full window."""
        result = sequence_tools.get_neighbors_window_of3(7, "ACDEFGHIK")
        assert result == "HIK"

    def test_short_sequence(self):
        """Very short sequences."""
        # Length 2 sequence
        assert sequence_tools.get_neighbors_window_of3(0, "AB") == "AB"
        assert sequence_tools.get_neighbors_window_of3(1, "AB") == "AB"
        
        # Length 3 sequence
        assert sequence_tools.get_neighbors_window_of3(1, "ABC") == "ABC"

    def test_window_length(self):
        """Window is at most 3 residues."""
        seq = "ACDEFGHIK"
        for i in range(len(seq)):
            window = sequence_tools.get_neighbors_window_of3(i, seq)
            assert len(window) <= 3
            assert len(window) >= 2  # At least 2 for boundary positions


# =============================================================================
# Tests for extract_fragments
# =============================================================================

class TestExtractFragments:
    """Tests for the fragment extraction from binary masks."""

    def test_docstring_example(self):
        """Example from docstring."""
        mask = [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
        result = sequence_tools.extract_fragments(mask, max_gap=1)
        assert result == ['111011', '1', '11', '101101']

    def test_gap_of_2_splits(self):
        """Gap of 2 zeros with max_gap=1 should split."""
        mask = [1, 1, 0, 0, 1, 1]
        result = sequence_tools.extract_fragments(mask, max_gap=1)
        assert result == ['11', '11']

    def test_gap_of_2_allowed(self):
        """Gap of 2 zeros with max_gap=2 should stay together."""
        mask = [1, 1, 0, 0, 1, 1]
        result = sequence_tools.extract_fragments(mask, max_gap=2)
        assert result == ['110011']

    def test_max_gap_0(self):
        """max_gap=0 means only consecutive 1s stay together."""
        mask = [1, 1, 0, 1, 1]
        result = sequence_tools.extract_fragments(mask, max_gap=0)
        assert result == ['11', '11']

    def test_all_zeros(self):
        """All zeros returns empty list."""
        result = sequence_tools.extract_fragments([0, 0, 0, 0])
        assert result == []

    def test_all_ones(self):
        """All ones returns single fragment."""
        result = sequence_tools.extract_fragments([1, 1, 1, 1])
        assert result == ['1111']

    def test_single_one(self):
        """Single 1 in mask."""
        result = sequence_tools.extract_fragments([0, 0, 1, 0, 0])
        assert result == ['1']

    def test_leading_trailing_zeros(self):
        """Leading and trailing zeros are stripped."""
        result = sequence_tools.extract_fragments([0, 0, 1, 1, 0, 0])
        assert result == ['11']

    def test_empty_mask(self):
        """Empty mask returns empty list."""
        result = sequence_tools.extract_fragments([])
        assert result == []


# =============================================================================
# Tests for count_nearby_hits
# =============================================================================

class TestCountNearbyHits:
    """Tests for counting nearby hits in a binary mask."""

    def test_simple_cluster(self):
        """Simple contiguous cluster of 1s."""
        result = sequence_tools.count_nearby_hits([0, 0, 1, 1, 1, 0, 0])
        assert result == [0, 0, 3, 3, 3, 0, 0]

    def test_separate_clusters(self):
        """Two separate clusters (gap > max_gap)."""
        result = sequence_tools.count_nearby_hits([1, 0, 0, 0, 1])
        assert result == [1, 0, 0, 0, 1]

    def test_connected_with_gap(self):
        """Clusters connected by allowed gap."""
        # With max_gap=1, a single 0 keeps them connected
        result = sequence_tools.count_nearby_hits([1, 0, 1], max_gap=1)
        assert result == [2, 0, 2]

    def test_window_size_limits(self):
        """Window size limits the count."""
        # Long run of 1s, but window_size=2 limits what we see
        mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 10 ones
        result = sequence_tools.count_nearby_hits(mask, window_size=2)
        # Position 0: sees positions 0, 1, 2 = 3
        # Position 5: sees positions 3, 4, 5, 6, 7 = 5
        assert result[0] == 3
        assert result[5] == 5

    def test_all_zeros(self):
        """All zeros returns all zeros."""
        result = sequence_tools.count_nearby_hits([0, 0, 0, 0])
        assert result == [0, 0, 0, 0]

    def test_all_ones(self):
        """All ones with small window."""
        result = sequence_tools.count_nearby_hits([1, 1, 1], window_size=1)
        # Position 0: sees 0, 1 = 2
        # Position 1: sees 0, 1, 2 = 3
        # Position 2: sees 1, 2 = 2
        assert result == [2, 3, 2]

    def test_empty_mask(self):
        """Empty mask returns empty list."""
        result = sequence_tools.count_nearby_hits([])
        assert result == []

    def test_preserves_length(self):
        """Output length matches input length."""
        mask = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
        result = sequence_tools.count_nearby_hits(mask)
        assert len(result) == len(mask)

    def test_zeros_stay_zero(self):
        """Positions with 0 in input stay 0 in output."""
        mask = [0, 1, 0, 1, 0]
        result = sequence_tools.count_nearby_hits(mask)
        for i, val in enumerate(mask):
            if val == 0:
                assert result[i] == 0


# =============================================================================
# Tests for show_sequence_HTML (basic functionality only)
# =============================================================================

class TestShowSequenceHTML:
    """Tests for the HTML sequence display function."""

    def test_returns_string(self):
        """Should return an HTML string when return_raw_string=True."""
        result = sequence_tools.show_sequence_HTML("ACDEF", return_raw_string=True)
        assert isinstance(result, str)

    def test_contains_residues(self):
        """Output should contain all residues from input."""
        seq = "ACDEF"
        result = sequence_tools.show_sequence_HTML(seq, return_raw_string=True)
        for residue in seq:
            assert residue in result

    def test_html_tags(self):
        """Output should contain HTML tags."""
        result = sequence_tools.show_sequence_HTML("ACDEF", return_raw_string=True)
        assert "<p" in result
        assert "</p>" in result
        assert "<span" in result

    def test_custom_colors(self):
        """Custom colors should be applied."""
        result = sequence_tools.show_sequence_HTML(
            "AAA", 
            colors={'A': '#ff0000'},
            return_raw_string=True
        )
        assert "#ff0000" in result

    def test_font_settings(self):
        """Font settings should be in output."""
        result = sequence_tools.show_sequence_HTML(
            "ACDEF",
            fontsize=20,
            font_family="Arial",
            return_raw_string=True
        )
        assert "20px" in result
        assert "Arial" in result

    def test_header(self):
        """Header should be included if provided."""
        result = sequence_tools.show_sequence_HTML(
            "ACDEF",
            header=">test_protein",
            return_raw_string=True
        )
        assert "test_protein" in result

    def test_blocksize_negative_one(self):
        """blocksize=-1 should use full sequence length."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        result = sequence_tools.show_sequence_HTML(seq, blocksize=-1, return_raw_string=True)
        # Should not have spaces from blocking
        assert isinstance(result, str)


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_mask_and_extract(self):
        """Mask a sequence then extract fragments."""
        seq = "KKAAAAKKAAAAAKK"
        # Mask for K residues
        mask = sequence_tools.mask_sequence(seq, ['K'])
        assert mask == [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]
        
        # Extract fragments (gap of 4-5 zeros should split)
        fragments = sequence_tools.extract_fragments(mask, max_gap=1)
        assert fragments == ['11', '11', '11']

    def test_mask_and_count(self):
        """Mask a sequence then count nearby hits."""
        seq = "AAKAA"
        mask = sequence_tools.mask_sequence(seq, ['K'])
        assert mask == [0, 0, 1, 0, 0]
        
        counts = sequence_tools.count_nearby_hits(mask)
        assert counts == [0, 0, 1, 0, 0]  # Single K sees only itself

    def test_charge_analysis_workflow(self):
        """Typical workflow: analyze charges in a sequence."""
        # M-K-D-K-E-K-R-K-A-K = 10 residues
        # Charged: K(5) + R(1) + D(1) + E(1) = 8 charged
        # Positive: K(5) + R(1) = 6
        # Negative: D(1) + E(1) = 2
        seq = "MKDKEKRKAK"
        
        fcr = sequence_tools.calculate_FCR(seq)
        ncpr = sequence_tools.calculate_NCPR(seq)
        
        # 8 charged residues out of 10
        assert fcr == pytest.approx(0.8)
        
        # Net charge: 6 positive - 2 negative = 4
        # NCPR = 4/10 = 0.4
        assert ncpr == pytest.approx(0.4)
