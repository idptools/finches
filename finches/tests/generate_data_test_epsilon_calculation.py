"""
Script to generate test data files for test_epsilon_calculation.py

Run this script from the tests directory to create the .npz files in test_data/
for both Mpipi_GGv1 and CALVADOS2 forcefields.

This script generates a comprehensive set of test sequences that vary in:
- Length (short, medium, long)
- Charge composition (positive, negative, neutral, mixed)
- Aliphatic content (high, low, clustered)
- Aromatic content
- Proline and glycine content
- Various biologically-relevant motifs
"""

import os
import numpy as np

from finches import epsilon_stateless as epsilon_calculation
from finches.epsilon_calculation import InteractionMatrixConstructor
from finches.forcefields.mpipi import Mpipi_model
from finches.forcefields.calvados import calvados_model


# =============================================================================
# COMPREHENSIVE TEST SEQUENCES
# =============================================================================
# Organized by category to test different aspects of epsilon calculation

TEST_SEQUENCES = {
    # -------------------------------------------------------------------------
    # LENGTH VARIATION
    # -------------------------------------------------------------------------
    # Very short sequences (edge cases)
    "short_5aa": "AEKLS",
    "short_10aa": "AEKLSQPGWY",
    "short_15aa": "AEKLSQPGWYFVMNH",
    
    # Medium sequences
    "medium_30aa": "MSKGEELFTGVVPILVELDGDVNGHKFSVS",
    "medium_50aa": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTG",
    
    # Longer sequences  
    "long_100aa": (
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLP"
        "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQT"
    ),
    "long_150aa": (
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLP"
        "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQLTLPLPN"
        "AEKLSQPGWYFVMNHAEKLSQPGWYFVMNHAEKLSQPGWYFVMNH"
    ),
    
    # -------------------------------------------------------------------------
    # CHARGE COMPOSITION
    # -------------------------------------------------------------------------
    # All positive (K, R)
    "all_positive_K": "KKKKKKKKKKKKKKKKKKKKKKKKKKKKKK",
    "all_positive_R": "RRRRRRRRRRRRRRRRRRRRRRRRRRRRRR",
    "all_positive_mixed": "KRKRKRKRKRKRKRKRKRKRKRKRKRKRKR",
    
    # All negative (E, D)
    "all_negative_E": "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
    "all_negative_D": "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDD",
    "all_negative_mixed": "DEDEDEDEDEDEDEDEDEDEDEDEDEDEDE",
    
    # Alternating charge (should have strong charge repulsion reduction)
    "alternating_EK": "EKEKEKEKEKEKEKEKEKEKEKEKEKEKEK",
    "alternating_DR": "DRDRDRDRDRDRDRDRDRDRDRDRDRDRD",
    "alternating_mixed": "EKDREKDREKDREKDREKDREKDREKDREK",
    
    # Charge blocks (clusters of same charge)
    "charge_blocks_10": "EEEEEKKKKKEEEEEKKKKKEEEEEKKKK",
    "charge_blocks_5": "EEEEEKKKKKEEEEEKKKKKEEEEEKKKKK",
    "charge_asymmetric": "EEEEEEEEEEEEEEEEEEEEKKKKKKKKK",
    
    # Net positive
    "net_positive": "KKKKKKKKKKEKEKEKEAAAAAAAAAAAA",
    
    # Net negative  
    "net_negative": "EEEEEEEEEEEKEKEKEAAAAAAAAAAAA",
    
    # -------------------------------------------------------------------------
    # ALIPHATIC COMPOSITION
    # -------------------------------------------------------------------------
    # High aliphatic (L, I, V, A, M)
    "aliphatic_L": "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
    "aliphatic_I": "IIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
    "aliphatic_V": "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVV",
    "aliphatic_A": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "aliphatic_mixed": "LIVAMLIVALIVAMLIVAMLIVALIVAMLI",
    
    # Aliphatic with charged residues
    "aliphatic_plus_charged": "LLLLLEEEEEKKKKKLLLLLEEEEEKKKK",
    
    # Aliphatic clusters
    "aliphatic_clusters": "LLLLLAAAAAEKEKEKLLLLLAAAAAEKEK",
    
    # Low aliphatic
    "low_aliphatic": "STQNSTQNSTQNSTQNSTQNSTQNSTQNST",
    
    # -------------------------------------------------------------------------
    # AROMATIC COMPOSITION
    # -------------------------------------------------------------------------
    # Aromatic rich (F, Y, W)
    "aromatic_F": "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
    "aromatic_Y": "YYYYYYYYYYYYYYYYYYYYYYYYYYYYYY",
    "aromatic_W": "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
    "aromatic_mixed": "FYWFYWFYWFYWFYWFYWFYWFYWFYWFYW",
    
    # Aromatic with polar
    "aromatic_polar": "FYSQFYSQFYSQFYSQFYSQFYSQFYSQFY",
    
    # Aromatic with charged (pi-cation potential)
    "aromatic_positive": "FYKFYKFYKFYKFYKFYKFYKFYKFYKFYK",
    "aromatic_negative": "FYEFYEFYEFYEFYEFYEFYEFYEFYEFYE",
    
    # -------------------------------------------------------------------------
    # PROLINE AND GLYCINE (FLEXIBILITY)
    # -------------------------------------------------------------------------
    # Proline rich (stiff)
    "proline_rich": "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPP",
    "proline_mixed": "PASPASPASPASPASPASPASPASPASPAS",
    
    # Glycine rich (flexible)
    "glycine_rich": "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
    "glycine_serine": "GSGSGSGSGSGSGSGSGSGSGSGSGSGSGS",
    
    # GS linker (common in constructs)
    "gs_linker_short": "GGGGSGGGGSGGGGSGGGGS",
    "gs_linker_long": "GGGGSGGGGSGGGGSGGGGSGGGGSGGGGSGGGGSGGGGSGGGGS",
    
    # -------------------------------------------------------------------------
    # POLAR/UNCHARGED
    # -------------------------------------------------------------------------
    # Serine/threonine rich
    "ser_thr_rich": "STSTSTSTSTSTSTSTSTSTSTSTSTSTST",
    
    # Asparagine/glutamine rich (Q/N rich - common in prions)
    "qn_rich": "QNQNQNQNQNQNQNQNQNQNQNQNQNQNQN",
    "q_rich": "QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ",
    "n_rich": "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNN",
    
    # -------------------------------------------------------------------------
    # CYSTEINE (SPECIAL RESIDUE)
    # -------------------------------------------------------------------------
    "cysteine_rich": "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "cysteine_spaced": "CAAAAACAAAAACAAAAACAAAAACAAAAA",
    
    # -------------------------------------------------------------------------
    # HISTIDINE (pH-DEPENDENT CHARGE)
    # -------------------------------------------------------------------------
    "histidine_rich": "HHHHHHHHHHHHHHHHHHHHHHHHHHHHHH",
    "histidine_with_E": "HEHEHEHEHEHEHEHEHEHEHEHEHEHEHE",
    "histidine_with_K": "HKHKHKHKHKHKHKHKHKHKHKHKHKHKHK",
    
    # -------------------------------------------------------------------------
    # BIOLOGICALLY-INSPIRED MOTIFS
    # -------------------------------------------------------------------------
    # RGG motif (common in RNA-binding proteins)
    "rgg_motif": "RGGRGGRGGRGGRGGRGGRGGRGGRGGRGGRGG",
    
    # FG repeats (nucleoporin-like)
    "fg_repeats": "FGFGFGFGFGFGFGFGFGFGFGFGFGFGFG",
    
    # Polyglutamate (common modification)
    "polyglutamate": "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEE",
    
    # SH3 binding motif pattern
    "pxxp_motif": "PXXPXXPXXPXXPXXPXXPXXPXXPXXPXX".replace("X", "A"),
    
    # Low complexity aromatic
    "low_complexity_fy": "FYFYFYFYFYFYFYFYFYFYFYFYFYFYFY",
    
    # -------------------------------------------------------------------------
    # MIXED COMPOSITION (REALISTIC)
    # -------------------------------------------------------------------------
    # Disordered region-like
    "idr_like_1": "SSQPSQSQPQSQSQPASPASQSQPQSQSPQ",
    "idr_like_2": "EKEKEKEAAAALLLLLSSSSSSQQQQQPPP",
    
    # Folded domain-like (balanced)
    "folded_like": "MVLSPADKTNVKAAWGKVGAHAGEYGAEAL",
    
    # Amphipathic (hydrophobic/hydrophilic alternating)
    "amphipathic": "LELALELALELALELALELALELELALEL",
    
    # Random realistic
    "realistic_mix1": "MSKGEELFTGVVPILVELDGDVNGHKFSVS",
    "realistic_mix2": "LLLLLAAAAAEKEKEKEAAAALELLLYYYYY",
    "realistic_mix3": "SSSSSSSQSQSQPSQPLSLLQSQAEKEKEK",
    
    # -------------------------------------------------------------------------
    # EDGE CASES
    # -------------------------------------------------------------------------
    # Single residue type but varied length
    "single_aa_A_20": "AAAAAAAAAAAAAAAAAAAA",
    "single_aa_E_20": "EEEEEEEEEEEEEEEEEEEE",
    "single_aa_L_20": "LLLLLLLLLLLLLLLLLLLL",
    "single_aa_K_20": "KKKKKKKKKKKKKKKKKKKK",
    
    # Exactly window size (31)
    "exact_window_31": "AEKLSQPGWYFVMNHAEKLSQPGWYFVMNH" + "A",
    
    # Just over window size
    "over_window_35": "AEKLSQPGWYFVMNHAEKLSQPGWYFVMNHAEKLS",

    # -------------------------------------------------------------------------
    # NATURAL SEQUENCES 
    # -------------------------------------------------------------------------
    "FUS" : "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGNYGQDQSSMSSGGGSGGGYGNQDQSGGGGSGGYGQQDRGGRGRGGSGGGGGGGGGGYNRSSGGYEPRGRGGGRGGRGGMGGSDRGGFNKFGGPRDQGSRHDSEQDNSDNNTIFVQGLGENVTIESVADYFKQIGIIKTNKKTGQPMINLYTDRETGKLKGEATVSFDDPPSAKAAIDWFDGKEFSGNPIKVSFATRRADFNRGGGNGRGGRGRGGPMGRGGYGGGGSGGGGRGGFPSGGGGGGGQQRAGDWKCPNPTCENMNFSWRNECNQCKAPKPDGPGGGPGGSHMGGNYGDDRRGGRGGYDRGGYRGRGGDRGGFRGGRGGGDRGGFGPGKMDSRGEHRQDRRERPY",
    "hnRNP_A1": "MSKSESPKEPEQLRKLFIGGLSFETTDESLRSHFEQWGTLTDCVVMRDPNTKRSRGFGFVTYATVEEVDAAMNARPHKVDGRVVEPKRAVSREDSQRPGAHLTVKKIFVGGIKEDTEEHHLRDYFEQYGKIEVIEIMTDRGSGKKRGFAFVTFDDHDSVDKIVIQKYHTVNGHNCEVRKALSKQEMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF",
    "velo1":"MNTTAPPPENGQYSTNQPRPYFYAQPTAQLPFQNPWYLGQLYNPYCIPGPGFRGGNPYFPYYSVALHEYPGYFVPQPQMNTRMSRRPHFNPHPPSPMFYHATRFRHYSSPGRRTETKETQTDPRQQECASKKQHSSDGKGCDGGNVVYLSSGISSTGNESNLENVEMSMSPATSTQERDFHKNACNSAQYRNMPPGSYAYEKEEVRIEYGSGSPAAIQMWKSYKETIPIYDVAVVKELPENVVQRDLFCEGVLYGPHAEGEELAVQSVAFSNKDECKNSLPPKLCIDAVQETETQTTIVQTREPRYETSKQGKQVMKVKATMEAESPTMVTEHVEVVSPVYDDPQVSVPEDSDEHNLITNGDLIEGSDGCPEQQDIANQSTCNGEVKLANKSNMWTDDSIEKFMPSPTWLACFENIDANYDYDVYSSQRKQKQTSVLSITSEELSSRDEGSSLDSASVSYFVPDYILRKGLYTFRKTTEDLEKETIKSSGSLKEDDIPLKQSCNKYVKKYRSSAVKAKDVSSRCRKIGVPLKGLSRRKLYSVKKNPKKSQSLSEPEDSDEYWVMEEENNEEGDDEDDSEEEEYYFQESLPHGQVDIGKGSIFKQIAQKRILWKPPKGMVPAQIVGWPVKEKLVTKKGAYDALNQVCRLKDYDGSDYTIYDKKISKLNRGFISEPKKSMQKSVGGKAQKKTPGTAVEEYWVGRGAKPKFPEPAYYLQDPTKIKEQDKPPKKKGALKSSKRKQTRTDPEEVETWEVPRSFLYRGHGLQKRGTKKKQLNGKLKPKKDKKKKADKQKQKEK",
    "pab1":"MKGNFRKRDSSTNSRKGGNSDSNYTNGGVPNQNNSSMFYENPEITRNFDDRQDYLLANSIGSDVTVTVTSGVKYTGLLVSCNLESTNGIDVVLRFPRVADSGVSDSVDDLAKTLGETLLIHGEDVAELELKNIDLSLDEKWENSKAQETTPARTNIEKERVNGESNEVTKFRTDVDISGSGREIKERKLEKWTPEEGAEHFDINKGKALEDDSASWDQFAVNEKKFGVKSTFDEHLYTTKINKDDPNYSKRLQEAERIAKEIESQGTSGNIHIAEDRGIIIDDSGLDEEDLYSGVDRRGDELLAALKSNSKPNSNKGNRYVPPTLRQQPHHMDPAIISSSNSNKNENAVSTDTSTPAAAGAPEGKPPQKTSKNKKSLSSKEAQIEELKKFSEKFKVPYDIPKDMLEVLKRSSSTLKSNSSLPPKPISKTPSAKTVSPTTQISAGKSESRRSGSNISQGQSSTGHTTRSSTSLRRRNHGSFFGAKNPHTNDAKRVLFGKSFNMFIKSKEAHDEKKKGDDASENMEPFFIEKPYFTAPTWLNTIEESYKTFFPDEDTAIQEAQTRFQQRQLNSMGNAVPGMNPAMGMNMGGMMGFPMGGPSASPNPMMNGFAAGSMGMYMPFQPQPMFYHPSMPQMMPVMGSNGAEEGGGNISPHVPAGFMAAGPGAPMGAFGYPGGIPFQGMMGSGPSGMPANGSAMHSHGHSRNYHQTSHHGHHNSSTSGHK"
}

# Create name list from dictionary keys
NAMES = list(TEST_SEQUENCES.keys())

# Convert to list for backward compatibility
TEST_SEQUENCES_LIST = list(TEST_SEQUENCES.values())


def generate_data_for_model(X, prefix):
    """Generate all test data for a given model."""
    
    print(f"\nGenerating data for {prefix}...")
    print(f"  Processing {len(NAMES)} sequences...")
    
    epsilon_vectors = {}
    epsilon_values = {}
    
    for name, seq in TEST_SEQUENCES.items():
        # NOWEIGHTING vectors
        try:
            attr_vec, rep_vec = epsilon_calculation.get_sequence_epsilon_vectors(
                seq, seq, X, use_charge_weighting=False, use_aliphatic_weighting=False)
            epsilon_vectors[f"{name}_NOWEIGHTING"] = [attr_vec, rep_vec]
        except Exception as e:
            print(f"  WARNING: Failed NOWEIGHTING for {name}: {e}")
            continue
        
        # Vectors with custom charge prefactor
        try:
            attr_vec, rep_vec = epsilon_calculation.get_sequence_epsilon_vectors(
                seq, seq, X, charge_prefactor=0.25)
            epsilon_vectors[f"{name}_charge_prefactor_25"] = [attr_vec, rep_vec]
        except Exception as e:
            print(f"  WARNING: Failed charge_prefactor_25 for {name}: {e}")

        # Vectors with custom null interaction baseline
        try:
            attr_vec, rep_vec = epsilon_calculation.get_sequence_epsilon_vectors(
                seq, seq, X, null_interaction_baseline=-0.15)
            epsilon_vectors[f"{name}_null_baseline_neg15"] = [attr_vec, rep_vec]
        except Exception as e:
            print(f"  WARNING: Failed null_baseline_neg15 for {name}: {e}")

        # Epsilon values
        try:
            epsilon_values[f"{name}_DEFAULT"] = np.float32(epsilon_calculation.get_sequence_epsilon_value(seq, seq, X))
            epsilon_values[f"{name}_NOCHARGE"] = np.float32(epsilon_calculation.get_sequence_epsilon_value(
                seq, seq, X, use_charge_weighting=False))
            epsilon_values[f"{name}_NOALIPHATICS"] = np.float32(epsilon_calculation.get_sequence_epsilon_value(
                seq, seq, X, use_aliphatic_weighting=False))
        except Exception as e:
            print(f"  WARNING: Failed epsilon values for {name}: {e}")
    
    # Convert vectors to float32 for reduced file size
    for key in epsilon_vectors:
        epsilon_vectors[key] = [v.astype(np.float32) for v in epsilon_vectors[key]]
    
    np.savez(f"test_data/{prefix}_seq_epsilon_and_vectors.npz",
             epsilon_vectors=epsilon_vectors, epsilon_values=epsilon_values)
    print(f"  Saved: test_data/{prefix}_seq_epsilon_and_vectors.npz")
    
    # Generate sliding epsilon data (only for sequences long enough)
    print(f"  Generating sliding epsilon data...")
    sliding_data = {}
    
    for name, seq in TEST_SEQUENCES.items():
        seq_len = len(seq)
        
        # Window size 1 works for all sequences
        try:
            result, _, _ = X.calculate_sliding_epsilon(seq, seq, window_size=1)
            sliding_data[f"sliding_{name}_w1"] = result.astype(np.float32)
        except Exception as e:
            print(f"  WARNING: Failed sliding w1 for {name}: {e}")
        
        # Window size 15 requires length >= 15
        if seq_len >= 15:
            try:
                result, _, _ = X.calculate_sliding_epsilon(seq, seq, window_size=15)
                sliding_data[f"sliding_{name}_w15"] = result.astype(np.float32)
            except Exception as e:
                print(f"  WARNING: Failed sliding w15 for {name}: {e}")
        
        # Window size 31 requires length >= 31
        if seq_len >= 31:
            try:
                result, _, _ = X.calculate_sliding_epsilon(seq, seq, window_size=31)
                sliding_data[f"sliding_{name}_default"] = result.astype(np.float32)
            except Exception as e:
                print(f"  WARNING: Failed sliding default for {name}: {e}")
    
    np.savez(f"test_data/{prefix}_sliding_epsilon.npz", **sliding_data)
    print(f"  Saved: test_data/{prefix}_sliding_epsilon.npz")
    
    # Generate heterotypic interaction data (pairs of different sequences)
    print(f"  Generating heterotypic interaction data...")
    heterotypic_data = {}
    
    # generate all possible pairs of different sequences
    hetero_pairs = []
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            if i != j:
                hetero_pairs.append((NAMES[i], NAMES[j]))
    
    for name1, name2 in hetero_pairs:
        if name1 in TEST_SEQUENCES and name2 in TEST_SEQUENCES:
            seq1 = TEST_SEQUENCES[name1]
            seq2 = TEST_SEQUENCES[name2]
            key = f"{name1}_vs_{name2}"
            
            try:
                epsilon_val = epsilon_calculation.get_sequence_epsilon_value(seq1, seq2, X)
                heterotypic_data[f"{key}_epsilon"] = np.float32(epsilon_val)
                
                attr_vec, rep_vec = epsilon_calculation.get_sequence_epsilon_vectors(seq1, seq2, X)
                heterotypic_data[f"{key}_attr"] = attr_vec.astype(np.float32)
                heterotypic_data[f"{key}_rep"] = rep_vec.astype(np.float32)
            except Exception as e:
                print(f"  WARNING: Failed heterotypic for {key}: {e}")
    
    np.savez(f"test_data/{prefix}_heterotypic.npz", **heterotypic_data)
    print(f"  Saved: test_data/{prefix}_heterotypic.npz")


def generate_matrix_data(X, prefix):
    """Generate pairwise matrix test data."""
    
    print(f"  Generating matrix test data...")
    matrix_data = {}
    
    
    for name in NAMES:
        if name in TEST_SEQUENCES:
            seq = TEST_SEQUENCES[name]
            
            # Homotypic matrix
            try:
                matrix = X.calculate_pairwise_homotypic_matrix(seq)
                matrix_data[f"{name}_homotypic"] = matrix.astype(np.float32)
            except Exception as e:
                print(f"  WARNING: Failed homotypic matrix for {name}: {e}")
            
            # Weighted matrix
            try:
                w_matrix = X.calculate_weighted_pairwise_matrix(seq, seq)
                matrix_data[f"{name}_weighted"] = w_matrix.astype(np.float32)
            except Exception as e:
                print(f"  WARNING: Failed weighted matrix for {name}: {e}")
    
    # Heterotypic matrices
    hetero_matrix_pairs = [
        ("short_10aa", "medium_30aa"),
        ("all_positive_K", "all_negative_E"),
    ]
    
    for name1, name2 in hetero_matrix_pairs:
        if name1 in TEST_SEQUENCES and name2 in TEST_SEQUENCES:
            seq1 = TEST_SEQUENCES[name1]
            seq2 = TEST_SEQUENCES[name2]
            
            try:
                matrix = X.calculate_pairwise_heterotypic_matrix(seq1, seq2)
                matrix_data[f"{name1}_vs_{name2}_heterotypic"] = matrix.astype(np.float32)
            except Exception as e:
                print(f"  WARNING: Failed heterotypic matrix for {name1} vs {name2}: {e}")
    
    np.savez(f"test_data/{prefix}_matrices.npz", **matrix_data)
    print(f"  Saved: test_data/{prefix}_matrices.npz")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    os.makedirs("test_data", exist_ok=True)
    
    print("=" * 60)
    print("Generating comprehensive test data for epsilon_calculation")
    print("=" * 60)
    print(f"\nTotal sequences to test: {len(TEST_SEQUENCES)}")
    print(f"Sequence categories:")
    print(f"  - Length variation: 7 sequences")
    print(f"  - Charge composition: 14 sequences")
    print(f"  - Aliphatic composition: 8 sequences")
    print(f"  - Aromatic composition: 7 sequences")
    print(f"  - Proline/Glycine: 6 sequences")
    print(f"  - Polar/uncharged: 5 sequences")
    print(f"  - Cysteine: 2 sequences")
    print(f"  - Histidine: 3 sequences")
    print(f"  - Biological motifs: 5 sequences")
    print(f"  - Mixed/realistic: 6 sequences")
    print(f"  - Edge cases: 6 sequences")
    
    # Generate data for Mpipi_GGv1
    print("\n" + "-" * 60)
    mpipi_params = Mpipi_model(version="Mpipi_GGv1")
    X_mpipi = InteractionMatrixConstructor(parameters=mpipi_params)
    generate_data_for_model(X_mpipi, "Mpipi_GGv1")
    generate_matrix_data(X_mpipi, "Mpipi_GGv1")
    
    # Generate data for CALVADOS2
    print("\n" + "-" * 60)
    calvados_params = calvados_model(version="CALVADOS2")
    X_calvados = InteractionMatrixConstructor(parameters=calvados_params)
    generate_data_for_model(X_calvados, "CALVADOS2")
    generate_matrix_data(X_calvados, "CALVADOS2")
    
    print("\n" + "=" * 60)
    print("Test data generation complete!")
    print("=" * 60)
    
    # Print summary of generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir("test_data")):
        if f.endswith(".npz"):
            size = os.path.getsize(f"test_data/{f}") / 1024
            print(f"  {f}: {size:.1f} KB")


if __name__ == "__main__":
    main()
