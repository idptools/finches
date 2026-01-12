"""
Tests for epsilon_calculation module.

Tests run against both Mpipi_GGv1 and CALVADOS2 forcefields.
Uses comprehensive test sequences covering various lengths and compositions.
"""

import os
import numpy as np
import pytest

from finches import epsilon_stateless as epsilon_calculation
from finches.epsilon_calculation import InteractionMatrixConstructor
from finches.forcefields.mpipi import Mpipi_model
from finches.forcefields.calvados import calvados_model


# Change to test directory for data files
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# COMPREHENSIVE TEST SEQUENCES
# =============================================================================
# Must match generate_data_test_epsilon_calculation.py

TEST_SEQUENCES = {
    # Length variation
    "short_5aa": "AEKLS",
    "short_10aa": "AEKLSQPGWY",
    "short_15aa": "AEKLSQPGWYFVMNH",
    "medium_30aa": "MSKGEELFTGVVPILVELDGDVNGHKFSVS",
    "medium_50aa": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTG",
    "long_100aa": (
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLP"
        "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQT"
    ),
    "long_150aa": (
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLP"
        "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQLTLPLPN"
        "AEKLSQPGWYFVMNHAEKLSQPGWYFVMNHAEKLSQPGWYFVMNH"
    ),
    
    # Charge composition
    "all_positive_K": "KKKKKKKKKKKKKKKKKKKKKKKKKKKKKK",
    "all_positive_R": "RRRRRRRRRRRRRRRRRRRRRRRRRRRRRR",
    "all_positive_mixed": "KRKRKRKRKRKRKRKRKRKRKRKRKRKRKR",
    "all_negative_E": "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
    "all_negative_D": "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDD",
    "all_negative_mixed": "DEDEDEDEDEDEDEDEDEDEDEDEDEDEDE",
    "alternating_EK": "EKEKEKEKEKEKEKEKEKEKEKEKEKEKEK",
    "alternating_DR": "DRDRDRDRDRDRDRDRDRDRDRDRDRDRD",
    "alternating_mixed": "EKDREKDREKDREKDREKDREKDREKDREK",
    "charge_blocks_10": "EEEEEKKKKKEEEEEKKKKKEEEEEKKKK",
    "charge_blocks_5": "EEEEEKKKKKEEEEEKKKKKEEEEEKKKKK",
    "charge_asymmetric": "EEEEEEEEEEEEEEEEEEEEKKKKKKKKK",
    "net_positive": "KKKKKKKKKKEKEKEKEAAAAAAAAAAAA",
    "net_negative": "EEEEEEEEEEEKEKEKEAAAAAAAAAAAA",
    
    # Aliphatic composition
    "aliphatic_L": "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
    "aliphatic_I": "IIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
    "aliphatic_V": "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
    "aliphatic_A": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "aliphatic_mixed": "LIVAMLIVALIVAMLIVAMLIVALIVAMLI",
    "aliphatic_plus_charged": "LLLLLEEEEEKKKKKLLLLLEEEEEKKKK",
    "aliphatic_clusters": "LLLLLAAAAAEKEKEKLLLLLAAAAAEKEK",
    "low_aliphatic": "STQNSTQNSTQNSTQNSTQNSTQNSTQNST",
    
    # Aromatic composition
    "aromatic_F": "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
    "aromatic_Y": "YYYYYYYYYYYYYYYYYYYYYYYYYYYYYY",
    "aromatic_W": "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
    "aromatic_mixed": "FYWFYWFYWFYWFYWFYWFYWFYWFYWFYW",
    "aromatic_polar": "FYSQFYSQFYSQFYSQFYSQFYSQFYSQFY",
    "aromatic_positive": "FYKFYKFYKFYKFYKFYKFYKFYKFYKFYK",
    "aromatic_negative": "FYEFYEFYEFYEFYEFYEFYEFYEFYEFYE",
    
    # Proline and Glycine
    "proline_rich": "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPP",
    "proline_mixed": "PASPASPASPASPASPASPASPASPASPAS",
    "glycine_rich": "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
    "glycine_serine": "GSGSGSGSGSGSGSGSGSGSGSGSGSGSGS",
    "gs_linker_short": "GGGGSGGGGSGGGGSGGGGS",
    "gs_linker_long": "GGGGSGGGGSGGGGSGGGGSGGGGSGGGGSGGGGSGGGGSGGGGS",
    
    # Polar/uncharged
    "ser_thr_rich": "STSTSTSTSTSTSTSTSTSTSTSTSTSTST",
    "qn_rich": "QNQNQNQNQNQNQNQNQNQNQNQNQNQNQN",
    "q_rich": "QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ",
    "n_rich": "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNN",
    
    # Cysteine
    "cysteine_rich": "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "cysteine_spaced": "CAAAAACAAAAACAAAAACAAAAACAAAAA",
    
    # Histidine
    "histidine_rich": "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHH",
    "histidine_with_E": "HEHEHEHEHEHEHEHEHEHEHEHEHEHEHE",
    "histidine_with_K": "HKHKHKHKHKHKHKHKHKHKHKHKHKHKHK",
    
    # Biological motifs
    "rgg_motif": "RGGRGGRGGRGGRGGRGGRGGRGGRGGRGGRGG",
    "fg_repeats": "FGFGFGFGFGFGFGFGFGFGFGFGFGFGFG",
    "polyglutamate": "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
    "pxxp_motif": "PXXPXXPXXPXXPXXPXXPXXPXXPXXPXX".replace("X", "A"),
    "low_complexity_fy": "FYFYFYFYFYFYFYFYFYFYFYFYFYFYFY",
    
    # Mixed/realistic
    "idr_like_1": "SSQPSQSQPQSQSQPASPASQSQPQSQSPQ",
    "idr_like_2": "EKEKEKEAAAALLLLLSSSSSSQQQQQPPP",
    "folded_like": "MVLSPADKTNVKAAWGKVGAHAGEYGAEAL",
    "amphipathic": "LELALELALELALELALELALELELALEL",
    "realistic_mix1": "MSKGEELFTGVVPILVELDGDVNGHKFSVS",
    "realistic_mix2": "LLLLLAAAAAEKEKEKEAAAALELLLYYYYY",
    "realistic_mix3": "SSSSSSSQSQSQPSQPLSLLQSQAEKEKEK",
    
    # Edge cases
    "single_aa_A_20": "AAAAAAAAAAAAAAAAAAAA",
    "single_aa_E_20": "EEEEEEEEEEEEEEEEEEEE",
    "single_aa_L_20": "LLLLLLLLLLLLLLLLLLLL",
    "single_aa_K_20": "KKKKKKKKKKKKKKKKKKKK",
    "exact_window_31": "AEKLSQPGWYFVMNHAEKLSQPGWYFVMNH" + "A",
    "over_window_35": "AEKLSQPGWYFVMNHAEKLSQPGWYFVMNHAEKLS",

    # Natural sequences
    "FUS" : "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGNYGQDQSSMSSGGGSGGGYGNQDQSGGGGSGGYGQQDRGGRGRGGSGGGGGGGGGGYNRSSGGYEPRGRGGGRGGRGGMGGSDRGGFNKFGGPRDQGSRHDSEQDNSDNNTIFVQGLGENVTIESVADYFKQIGIIKTNKKTGQPMINLYTDRETGKLKGEATVSFDDPPSAKAAIDWFDGKEFSGNPIKVSFATRRADFNRGGGNGRGGRGRGGPMGRGGYGGGGSGGGGRGGFPSGGGGGGGQQRAGDWKCPNPTCENMNFSWRNECNQCKAPKPDGPGGGPGGSHMGGNYGDDRRGGRGGYDRGGYRGRGGDRGGFRGGRGGGDRGGFGPGKMDSRGEHRQDRRERPY",
    "hnRNP_A1": "MSKSESPKEPEQLRKLFIGGLSFETTDESLRSHFEQWGTLTDCVVMRDPNTKRSRGFGFVTYATVEEVDAAMNARPHKVDGRVVEPKRAVSREDSQRPGAHLTVKKIFVGGIKEDTEEHHLRDYFEQYGKIEVIEIMTDRGSGKKRGFAFVTFDDHDSVDKIVIQKYHTVNGHNCEVRKALSKQEMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF",
    "velo1":"MNTTAPPPENGQYSTNQPRPYFYAQPTAQLPFQNPWYLGQLYNPYCIPGPGFRGGNPYFPYYSVALHEYPGYFVPQPQMNTRMSRRPHFNPHPPSPMFYHATRFRHYSSPGRRTETKETQTDPRQQECASKKQHSSDGKGCDGGNVVYLSSGISSTGNESNLENVEMSMSPATSTQERDFHKNACNSAQYRNMPPGSYAYEKEEVRIEYGSGSPAAIQMWKSYKETIPIYDVAVVKELPENVVQRDLFCEGVLYGPHAEGEELAVQSVAFSNKDECKNSLPPKLCIDAVQETETQTTIVQTREPRYETSKQGKQVMKVKATMEAESPTMVTEHVEVVSPVYDDPQVSVPEDSDEHNLITNGDLIEGSDGCPEQQDIANQSTCNGEVKLANKSNMWTDDSIEKFMPSPTWLACFENIDANYDYDVYSSQRKQKQTSVLSITSEELSSRDEGSSLDSASVSYFVPDYILRKGLYTFRKTTEDLEKETIKSSGSLKEDDIPLKQSCNKYVKKYRSSAVKAKDVSSRCRKIGVPLKGLSRRKLYSVKKNPKKSQSLSEPEDSDEYWVMEEENNEEGDDEDDSEEEEYYFQESLPHGQVDIGKGSIFKQIAQKRILWKPPKGMVPAQIVGWPVKEKLVTKKGAYDALNQVCRLKDYDGSDYTIYDKKISKLNRGFISEPKKSMQKSVGGKAQKKTPGTAVEEYWVGRGAKPKFPEPAYYLQDPTKIKEQDKPPKKKGALKSSKRKQTRTDPEEVETWEVPRSFLYRGHGLQKRGTKKKQLNGKLKPKKDKKKKADKQKQKEK",
    "pab1":"MKGNFRKRDSSTNSRKGGNSDSNYTNGGVPNQNNSSMFYENPEITRNFDDRQDYLLANSIGSDVTVTVTSGVKYTGLLVSCNLESTNGIDVVLRFPRVADSGVSDSVDDLAKTLGETLLIHGEDVAELELKNIDLSLDEKWENSKAQETTPARTNIEKERVNGESNEVTKFRTDVDISGSGREIKERKLEKWTPEEGAEHFDINKGKALEDDSASWDQFAVNEKKFGVKSTFDEHLYTTKINKDDPNYSKRLQEAERIAKEIESQGTSGNIHIAEDRGIIIDDSGLDEEDLYSGVDRRGDELLAALKSNSKPNSNKGNRYVPPTLRQQPHHMDPAIISSSNSNKNENAVSTDTSTPAAAGAPEGKPPQKTSKNKKSLSSKEAQIEELKKFSEKFKVPYDIPKDMLEVLKRSSSTLKSNSSLPPKPISKTPSAKTVSPTTQISAGKSESRRSGSNISQGQSSTGHTTRSSTSLRRRNHGSFFGAKNPHTNDAKRVLFGKSFNMFIKSKEAHDEKKKGDDASENMEPFFIEKPYFTAPTWLNTIEESYKTFFPDEDTAIQEAQTRFQQRQLNSMGNAVPGMNPAMGMNMGGMMGFPMGGPSASPNPMMNGFAAGSMGMYMPFQPQPMFYHPSMPQMMPVMGSNGAEEGGGNISPHVPAGFMAAGPGAPMGAFGYPGGIPFQGMMGSGPSGMPANGSAMHSHGHSRNYHQTSHHGHHNSSTSGHK"

}

NAMES = list(TEST_SEQUENCES.keys())


# ============================================================================
#                              FIXTURES
# ============================================================================

@pytest.fixture(params=["mpipi", "calvados"])
def model_fixture(request):
    """
    Fixture that provides both model types for parametrized testing.
    Returns tuple of (model_name, IMC_instance, data_file_prefix)
    """
    if request.param == "mpipi":
        params = Mpipi_model(version='Mpipi_GGv1')
        imc = InteractionMatrixConstructor(parameters=params)
        return ("mpipi", imc, "Mpipi_GGv1")
    else:
        params = calvados_model(version='CALVADOS2')
        imc = InteractionMatrixConstructor(parameters=params)
        return ("calvados", imc, "CALVADOS2")


@pytest.fixture
def mpipi_model():
    """Fixture for Mpipi model only (for tests that don't need both)."""
    params = Mpipi_model(version='Mpipi_GGv1')
    return InteractionMatrixConstructor(parameters=params)


@pytest.fixture
def calvados_model_fixture():
    """Fixture for CALVADOS model only."""
    params = calvados_model(version='CALVADOS2')
    return InteractionMatrixConstructor(parameters=params)


# ============================================================================
#                         PARAMETRIZED TESTS
# ============================================================================

def test_get_sequence_epsilon_vectors(model_fixture):
    """Test epsilon vectors calculation against saved truth data."""
    model_name, X, prefix = model_fixture
    
    TRUE_data = np.load(
        f"test_data/{prefix}_seq_epsilon_and_vectors.npz", allow_pickle=True
    )
    epsilon_vectors = TRUE_data["epsilon_vectors"].item()

    for name, seq in TEST_SEQUENCES.items():
        # Skip if this sequence wasn't in the generated data
        if f"{name}_NOWEIGHTING" not in epsilon_vectors:
            continue

        # compare vectors with no weighting
        [attractive_vector, repulsive_vector] = epsilon_vectors[f"{name}_NOWEIGHTING"]
        test_attr, test_rep = epsilon_calculation.get_sequence_epsilon_vectors(
            seq, seq, X,
            use_charge_weighting=False,
            use_aliphatic_weighting=False,
        )
        assert np.allclose(test_attr, attractive_vector), f"{model_name}: {name}_NOWEIGHTING attractive mismatch"
        assert np.allclose(test_rep, repulsive_vector), f"{model_name}: {name}_NOWEIGHTING repulsive mismatch"

        # compare vectors with custom charge prefactor
        if f"{name}_charge_prefactor_25" in epsilon_vectors:
            [attractive_vector, repulsive_vector] = epsilon_vectors[f"{name}_charge_prefactor_25"]
            test_attr, test_rep = epsilon_calculation.get_sequence_epsilon_vectors(
                seq, seq, X, charge_prefactor=0.25
            )
            assert np.allclose(test_attr, attractive_vector), f"{model_name}: {name}_charge_prefactor_25 attractive mismatch"
            assert np.allclose(test_rep, repulsive_vector), f"{model_name}: {name}_charge_prefactor_25 repulsive mismatch"

        # compare vectors with custom null interaction baseline
        if f"{name}_null_baseline_neg15" in epsilon_vectors:
            [attractive_vector, repulsive_vector] = epsilon_vectors[f"{name}_null_baseline_neg15"]
            test_attr, test_rep = epsilon_calculation.get_sequence_epsilon_vectors(
                seq, seq, X, null_interaction_baseline=-0.15
            )
            assert np.allclose(test_attr, attractive_vector), f"{model_name}: {name}_null_baseline_neg15 attractive mismatch"
            assert np.allclose(test_rep, repulsive_vector), f"{model_name}: {name}_null_baseline_neg15 repulsive mismatch"


def test_get_sequence_epsilon_value(model_fixture):
    """Test epsilon value calculation against saved truth data."""
    model_name, X, prefix = model_fixture
    
    TRUE_data = np.load(
        f"test_data/{prefix}_seq_epsilon_and_vectors.npz", allow_pickle=True
    )
    all_epsilon_values = TRUE_data["epsilon_values"].item()

    for name, seq in TEST_SEQUENCES.items():
        # Skip if this sequence wasn't in the generated data
        if f"{name}_DEFAULT" not in all_epsilon_values:
            continue

        # test default epsilon value
        TRUE_epsilon = all_epsilon_values[f"{name}_DEFAULT"]
        TEST_epsilon = epsilon_calculation.get_sequence_epsilon_value(seq, seq, X)
        assert np.isclose(TEST_epsilon, TRUE_epsilon), f"{model_name}: {name}_DEFAULT mismatch"

        # test epsilon with no charge weighting
        if f"{name}_NOCHARGE" in all_epsilon_values:
            TRUE_epsilon = all_epsilon_values[f"{name}_NOCHARGE"]
            TEST_epsilon = epsilon_calculation.get_sequence_epsilon_value(
                seq, seq, X, use_charge_weighting=False
            )
            assert np.isclose(TEST_epsilon, TRUE_epsilon), f"{model_name}: {name}_NOCHARGE mismatch"

        # test epsilon with no aliphatic weighting
        if f"{name}_NOALIPHATICS" in all_epsilon_values:
            TRUE_epsilon = all_epsilon_values[f"{name}_NOALIPHATICS"]
            TEST_epsilon = epsilon_calculation.get_sequence_epsilon_value(
                seq, seq, X, use_aliphatic_weighting=False
            )
            assert np.isclose(TEST_epsilon, TRUE_epsilon), f"{model_name}: {name}_NOALIPHATICS mismatch"


def test_calculate_sliding_epsilon(model_fixture):
    """Test sliding epsilon calculation against saved truth data."""
    model_name, X, prefix = model_fixture
    
    TRUE_data = np.load(
        f"test_data/{prefix}_sliding_epsilon.npz", allow_pickle=True
    )

    for name, seq in TEST_SEQUENCES.items():
        seq_len = len(seq)
        
        # test sliding epsilon with default window size (31) - only for long enough sequences
        if seq_len >= 31 and f"sliding_{name}_default" in TRUE_data:
            TRUE_sliding = TRUE_data[f"sliding_{name}_default"]
            TEST_sliding, _, _ = X.calculate_sliding_epsilon(seq, seq, window_size=31)
            assert np.allclose(TEST_sliding, TRUE_sliding), f"{model_name}: {name} default window mismatch"

        # test with window size 15 - only for sequences >= 15
        if seq_len >= 15 and f"sliding_{name}_w15" in TRUE_data:
            TRUE_sliding_w15 = TRUE_data[f"sliding_{name}_w15"]
            TEST_sliding_w15, _, _ = X.calculate_sliding_epsilon(seq, seq, window_size=15)
            assert np.allclose(TEST_sliding_w15, TRUE_sliding_w15), f"{model_name}: {name} w15 mismatch"

        # test with window size 1 - works for all sequences
        if f"sliding_{name}_w1" in TRUE_data:
            TRUE_sliding_w1 = TRUE_data[f"sliding_{name}_w1"]
            TEST_sliding_w1, _, _ = X.calculate_sliding_epsilon(seq, seq, window_size=1)
            assert np.allclose(TEST_sliding_w1, TRUE_sliding_w1), f"{model_name}: {name} w1 mismatch"


def test_calculate_sliding_epsilon_window_size_validation(model_fixture):
    """Test that even window sizes are rounded up to odd."""
    _, X, _ = model_fixture
    seq = TEST_SEQUENCES["medium_50aa"]  # Use a specific longer sequence
    
    result1, _, _ = X.calculate_sliding_epsilon(seq, seq, window_size=30)
    result2, _, _ = X.calculate_sliding_epsilon(seq, seq, window_size=31)
    assert np.allclose(result1, result2)


def test_calculate_sliding_epsilon_window_size_error(model_fixture):
    """Test that window size larger than sequence raises error."""
    _, X, _ = model_fixture
    short_seq = "AAAA"
    
    with pytest.raises(ValueError, match="Window size larger than matrix size"):
        X.calculate_sliding_epsilon(short_seq, short_seq, window_size=31, use_cython=False)


def test_check_sequence_empty(model_fixture):
    """Test that empty sequence raises error."""
    _, X, _ = model_fixture
    
    with pytest.raises(ValueError, match="Empty sequence provided"):
        X._check_sequence("")


def test_calculate_epsilon_vectors_method(model_fixture):
    """Test that instance method matches module function."""
    _, X, _ = model_fixture
    seq = TEST_SEQUENCES["medium_30aa"]
    
    attr_vec_method, rep_vec_method = X.calculate_epsilon_vectors(seq, seq)
    attr_vec_func, rep_vec_func = epsilon_calculation.get_sequence_epsilon_vectors(seq, seq, X)
    
    assert np.allclose(attr_vec_method, attr_vec_func)
    assert np.allclose(rep_vec_method, rep_vec_func)


def test_calculate_epsilon_value_method(model_fixture):
    """Test that instance method matches module function."""
    _, X, _ = model_fixture
    seq = TEST_SEQUENCES["medium_30aa"]
    
    eps_method = X.calculate_epsilon_value(seq, seq)
    eps_func = epsilon_calculation.get_sequence_epsilon_value(seq, seq, X)
    
    assert np.isclose(eps_method, eps_func)


def test_homotypic_vs_heterotypic_same_sequence(model_fixture):
    """Test that homotypic matrix equals heterotypic with same sequence."""
    _, X, _ = model_fixture
    seq = TEST_SEQUENCES["medium_50aa"]
    
    homotypic_matrix = X.calculate_pairwise_homotypic_matrix(seq)
    heterotypic_matrix = X.calculate_pairwise_heterotypic_matrix(seq, seq)
    
    assert np.allclose(homotypic_matrix, heterotypic_matrix)


def test_matrix_symmetry_homotypic(model_fixture):
    """Test that homotypic matrix is symmetric."""
    _, X, _ = model_fixture
    seq = TEST_SEQUENCES["medium_50aa"]
    
    matrix = X.calculate_pairwise_homotypic_matrix(seq)
    assert np.allclose(matrix, matrix.T)


def test_weighted_matrix_without_cython(model_fixture):
    """Test that cython and python implementations match for weighted matrix."""
    _, X, _ = model_fixture
    seq = TEST_SEQUENCES["medium_50aa"]
    
    matrix_cython = X.calculate_weighted_pairwise_matrix(seq, seq, use_cython=True)
    matrix_python = X.calculate_weighted_pairwise_matrix(seq, seq, use_cython=False)
    
    assert np.allclose(matrix_cython, matrix_python)


def test_heterotypic_matrix_without_cython(model_fixture):
    """Test that cython and python implementations match for heterotypic matrix."""
    _, X, _ = model_fixture
    seq = TEST_SEQUENCES["medium_50aa"]
    
    matrix_cython = X.calculate_pairwise_heterotypic_matrix(seq, seq, use_cython=True)
    matrix_python = X.calculate_pairwise_heterotypic_matrix(seq, seq, use_cython=False)
    
    assert np.allclose(matrix_cython, matrix_python)


def test_heterotypic_epsilon_values(model_fixture):
    """Test heterotypic epsilon values against saved truth data."""
    model_name, X, prefix = model_fixture
    
    TRUE_data = np.load(
        f"test_data/{prefix}_heterotypic.npz", allow_pickle=True
    )
    
    # Test a subset of heterotypic pairs to keep test time reasonable
    # while still covering different interaction types
    test_pairs = [
        # Charge-charge interactions
        ("all_positive_K", "all_negative_E"),
        ("all_positive_R", "all_negative_D"),
        ("alternating_EK", "alternating_DR"),
        # Aromatic-charged interactions (pi-cation)
        ("aromatic_F", "all_positive_K"),
        ("aromatic_Y", "all_positive_R"),
        ("aromatic_W", "all_negative_E"),
        # Aliphatic-aromatic interactions
        ("aliphatic_L", "aromatic_F"),
        ("aliphatic_mixed", "aromatic_mixed"),
        # Polar-charged interactions
        ("ser_thr_rich", "all_positive_K"),
        ("qn_rich", "all_negative_E"),
        # Length variations
        ("short_10aa", "medium_50aa"),
        ("medium_30aa", "long_100aa"),
        # Biological motifs
        ("rgg_motif", "aromatic_F"),
        ("fg_repeats", "glycine_rich"),
        # Realistic sequences
        ("idr_like_1", "folded_like"),
        ("realistic_mix1", "realistic_mix2"),
    ]
    
    for name1, name2 in test_pairs:
        key = f"{name1}_vs_{name2}"
        epsilon_key = f"{key}_epsilon"
        
        # Skip if this pair wasn't generated
        if epsilon_key not in TRUE_data:
            continue
        
        seq1 = TEST_SEQUENCES[name1]
        seq2 = TEST_SEQUENCES[name2]
        
        TRUE_epsilon = TRUE_data[epsilon_key]
        TEST_epsilon = epsilon_calculation.get_sequence_epsilon_value(seq1, seq2, X)
        
        assert np.isclose(TEST_epsilon, TRUE_epsilon), \
            f"{model_name}: {key} epsilon mismatch (expected {TRUE_epsilon}, got {TEST_epsilon})"


def test_heterotypic_epsilon_vectors(model_fixture):
    """Test heterotypic epsilon vectors against saved truth data."""
    model_name, X, prefix = model_fixture
    
    TRUE_data = np.load(
        f"test_data/{prefix}_heterotypic.npz", allow_pickle=True
    )
    
    # Test a representative subset of heterotypic pairs
    test_pairs = [
        ("all_positive_K", "all_negative_E"),
        ("aromatic_F", "all_positive_K"),
        ("aliphatic_L", "aromatic_F"),
        ("short_10aa", "medium_50aa"),
        ("rgg_motif", "aromatic_F"),
        ("idr_like_1", "folded_like"),
    ]
    
    for name1, name2 in test_pairs:
        key = f"{name1}_vs_{name2}"
        attr_key = f"{key}_attr"
        rep_key = f"{key}_rep"
        
        # Skip if this pair wasn't generated
        if attr_key not in TRUE_data or rep_key not in TRUE_data:
            continue
        
        seq1 = TEST_SEQUENCES[name1]
        seq2 = TEST_SEQUENCES[name2]
        
        TRUE_attr = TRUE_data[attr_key]
        TRUE_rep = TRUE_data[rep_key]
        
        TEST_attr, TEST_rep = epsilon_calculation.get_sequence_epsilon_vectors(seq1, seq2, X)
        
        assert np.allclose(TEST_attr, TRUE_attr), \
            f"{model_name}: {key} attractive vector mismatch"
        assert np.allclose(TEST_rep, TRUE_rep), \
            f"{model_name}: {key} repulsive vector mismatch"


def test_heterotypic_asymmetry(model_fixture):
    """Test heterotypic interactions for pairs of different length sequences.
    
    Epsilon values can be asymmetric (A->B != B->A) because epsilon is 
    normalized by the length of the first sequence. This test verifies that
    our calculations match the saved reference data for both directions.
    """
    model_name, X, prefix = model_fixture
    
    TRUE_data = np.load(
        f"test_data/{prefix}_heterotypic.npz", allow_pickle=True
    )
    
    # Test pairs where order matters (different length sequences)
    asymmetric_pairs = [
        ("short_10aa", "medium_50aa"),
        ("medium_30aa", "long_100aa"),
        ("short_5aa", "long_150aa"),
    ]
    
    for name1, name2 in asymmetric_pairs:
        key_forward = f"{name1}_vs_{name2}_epsilon"
        key_reverse = f"{name2}_vs_{name1}_epsilon"
        
        # Skip if pairs weren't generated
        if key_forward not in TRUE_data or key_reverse not in TRUE_data:
            continue
        
        seq1 = TEST_SEQUENCES[name1]
        seq2 = TEST_SEQUENCES[name2]
        
        # Get saved reference values
        TRUE_forward = TRUE_data[key_forward]
        TRUE_reverse = TRUE_data[key_reverse]
        
        # Calculate both directions
        TEST_forward = epsilon_calculation.get_sequence_epsilon_value(seq1, seq2, X)
        TEST_reverse = epsilon_calculation.get_sequence_epsilon_value(seq2, seq1, X)
        
        # Verify both match saved data
        assert np.isclose(TEST_forward, TRUE_forward), \
            f"{model_name}: {name1}_vs_{name2} forward epsilon mismatch"
        assert np.isclose(TEST_reverse, TRUE_reverse), \
            f"{model_name}: {name2}_vs_{name1} reverse epsilon mismatch"
        
        # For sequences of different lengths, epsilon IS expected to be 
        # asymmetric because it's normalized by the first sequence's length.
        # Verify this asymmetry exists in the data
        if len(seq1) != len(seq2):
            assert not np.isclose(TEST_forward, TEST_reverse), \
                f"{model_name}: expected asymmetry for different length sequences {name1} vs {name2}"


def test_heterotypic_vs_homotypic_different(model_fixture):
    """Test that heterotypic epsilon differs from homotypic for different sequences."""
    _, X, _ = model_fixture
    
    pairs = [
        ("all_positive_K", "all_negative_E"),
        ("aromatic_F", "aliphatic_L"),
        ("proline_rich", "glycine_rich"),
    ]
    
    for name1, name2 in pairs:
        seq1 = TEST_SEQUENCES[name1]
        seq2 = TEST_SEQUENCES[name2]
        
        eps_homotypic_1 = epsilon_calculation.get_sequence_epsilon_value(seq1, seq1, X)
        eps_homotypic_2 = epsilon_calculation.get_sequence_epsilon_value(seq2, seq2, X)
        eps_heterotypic = epsilon_calculation.get_sequence_epsilon_value(seq1, seq2, X)
        
        # Heterotypic should generally be different from both homotypic values
        # (not always true mathematically, but for these very different sequences it should be)
        # At minimum, it shouldn't equal both homotypic values
        both_equal = np.isclose(eps_heterotypic, eps_homotypic_1) and \
                     np.isclose(eps_heterotypic, eps_homotypic_2)
        assert not both_equal, \
            f"Heterotypic epsilon unexpectedly equals both homotypic values for {name1} vs {name2}"


# ============================================================================
#                         MATRIX TESTS
# ============================================================================

def test_homotypic_matrices(model_fixture):
    """Test homotypic matrices against saved truth data."""
    model_name, X, prefix = model_fixture
    
    TRUE_data = np.load(
        f"test_data/{prefix}_matrices.npz", allow_pickle=True
    )
    
    # Test a representative subset of sequences
    test_sequences = [
        "short_10aa",
        "medium_30aa", 
        "medium_50aa",
        "all_positive_K",
        "all_negative_E",
        "alternating_EK",
        "aliphatic_L",
        "aromatic_F",
        "aromatic_mixed",
        "proline_rich",
        "glycine_rich",
        "idr_like_1",
        "folded_like",
    ]
    
    for name in test_sequences:
        key = f"{name}_homotypic"
        
        # Skip if this sequence wasn't generated
        if key not in TRUE_data:
            continue
        
        seq = TEST_SEQUENCES[name]
        TRUE_matrix = TRUE_data[key]
        TEST_matrix = X.calculate_pairwise_homotypic_matrix(seq)
        
        assert np.allclose(TEST_matrix, TRUE_matrix), \
            f"{model_name}: {name} homotypic matrix mismatch"
        
        # Verify matrix dimensions match sequence length
        assert TEST_matrix.shape == (len(seq), len(seq)), \
            f"{model_name}: {name} matrix shape mismatch"


def test_weighted_matrices(model_fixture):
    """Test weighted pairwise matrices against saved truth data."""
    model_name, X, prefix = model_fixture
    
    TRUE_data = np.load(
        f"test_data/{prefix}_matrices.npz", allow_pickle=True
    )
    
    # Test sequences that benefit from weighting (charged, aliphatic)
    test_sequences = [
        "medium_50aa",
        "all_positive_K",
        "all_negative_E",
        "alternating_EK",
        "charge_blocks_10",
        "aliphatic_L",
        "aliphatic_mixed",
        "aliphatic_plus_charged",
        "aromatic_positive",
        "aromatic_negative",
        "idr_like_2",
    ]
    
    for name in test_sequences:
        key = f"{name}_weighted"
        
        # Skip if this sequence wasn't generated
        if key not in TRUE_data:
            continue
        
        seq = TEST_SEQUENCES[name]
        TRUE_matrix = TRUE_data[key]
        TEST_matrix = X.calculate_weighted_pairwise_matrix(seq, seq)
        
        assert np.allclose(TEST_matrix, TRUE_matrix), \
            f"{model_name}: {name} weighted matrix mismatch"


def test_heterotypic_matrices(model_fixture):
    """Test heterotypic matrices against saved truth data."""
    model_name, X, prefix = model_fixture
    
    TRUE_data = np.load(
        f"test_data/{prefix}_matrices.npz", allow_pickle=True
    )
    
    # Test the heterotypic pairs that were generated
    hetero_pairs = [
        ("short_10aa", "medium_30aa"),
        ("all_positive_K", "all_negative_E"),
    ]
    
    for name1, name2 in hetero_pairs:
        key = f"{name1}_vs_{name2}_heterotypic"
        
        # Skip if this pair wasn't generated
        if key not in TRUE_data:
            continue
        
        seq1 = TEST_SEQUENCES[name1]
        seq2 = TEST_SEQUENCES[name2]
        TRUE_matrix = TRUE_data[key]
        TEST_matrix = X.calculate_pairwise_heterotypic_matrix(seq1, seq2)
        
        assert np.allclose(TEST_matrix, TRUE_matrix), \
            f"{model_name}: {name1} vs {name2} heterotypic matrix mismatch"
        
        # Verify matrix dimensions match sequence lengths
        assert TEST_matrix.shape == (len(seq1), len(seq2)), \
            f"{model_name}: {name1} vs {name2} matrix shape mismatch (expected {(len(seq1), len(seq2))}, got {TEST_matrix.shape})"


def test_homotypic_matrix_diagonal(model_fixture):
    """Test that homotypic matrix diagonal represents self-interactions."""
    _, X, _ = model_fixture
    
    test_sequences = ["medium_30aa", "all_positive_K", "aromatic_F"]
    
    for name in test_sequences:
        seq = TEST_SEQUENCES[name]
        matrix = X.calculate_pairwise_homotypic_matrix(seq)
        
        # The diagonal should contain the self-interaction energies
        # which should all be finite values
        diagonal = np.diag(matrix)
        assert np.all(np.isfinite(diagonal)), \
            f"Non-finite values in {name} homotypic matrix diagonal"


def test_weighted_vs_unweighted_matrix_difference(model_fixture):
    """Test that weighted matrices differ from unweighted for charged sequences."""
    _, X, _ = model_fixture
    
    # For highly charged sequences, weighting should make a difference
    charged_sequences = ["all_positive_K", "all_negative_E", "alternating_EK"]
    
    for name in charged_sequences:
        seq = TEST_SEQUENCES[name]
        
        unweighted = X.calculate_pairwise_homotypic_matrix(seq)
        weighted = X.calculate_weighted_pairwise_matrix(seq, seq)
        
        # Weighted matrix should differ from unweighted for charged sequences
        # (charge weighting modifies the interaction values)
        assert not np.allclose(unweighted, weighted), \
            f"{name}: weighted matrix unexpectedly equals unweighted matrix"


def test_heterotypic_matrix_dimensions(model_fixture):
    """Test that heterotypic matrix has correct dimensions for different length sequences."""
    _, X, _ = model_fixture
    
    pairs = [
        ("short_5aa", "medium_30aa"),
        ("short_10aa", "long_100aa"),
        ("medium_50aa", "short_15aa"),
    ]
    
    for name1, name2 in pairs:
        seq1 = TEST_SEQUENCES[name1]
        seq2 = TEST_SEQUENCES[name2]
        
        matrix = X.calculate_pairwise_heterotypic_matrix(seq1, seq2)
        
        expected_shape = (len(seq1), len(seq2))
        assert matrix.shape == expected_shape, \
            f"Heterotypic matrix for {name1} vs {name2}: expected shape {expected_shape}, got {matrix.shape}"


def test_matrix_values_finite(model_fixture):
    """Test that all matrix values are finite (no NaN or Inf)."""
    _, X, _ = model_fixture
    
    test_sequences = [
        "cysteine_rich",
        "histidine_rich",
        "proline_rich",
        "glycine_rich",
        "aromatic_W",  # Tryptophan-rich
    ]
    
    for name in test_sequences:
        seq = TEST_SEQUENCES[name]
        
        homotypic = X.calculate_pairwise_homotypic_matrix(seq)
        weighted = X.calculate_weighted_pairwise_matrix(seq, seq)
        
        assert np.all(np.isfinite(homotypic)), \
            f"{name}: non-finite values in homotypic matrix"
        assert np.all(np.isfinite(weighted)), \
            f"{name}: non-finite values in weighted matrix"


# ============================================================================
#                    NON-PARAMETRIZED TESTS (run once)
# ============================================================================

def test_interaction_matrix_constructor_invalid_parameters():
    """Test that invalid parameters object raises error."""
    class InvalidParams:
        pass

    with pytest.raises(AttributeError):
        InteractionMatrixConstructor(InvalidParams())
