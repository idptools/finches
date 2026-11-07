# finches package data

This directory holds the forcefield parameter files and reference data that ship with finches. Paths are resolved at runtime with `finches.get_data(path)`, which returns the absolute path to a file inside this directory.

## Mpipi (`Mpipi/`)

Pairwise Mpipi parameters stored as pickled dictionaries, each indexed as `[residue_1][residue_2]` (or `[residue]` for the charge table). Three parameter sets are available and are selected through the `version` argument of `finches.forcefields.mpipi.Mpipi_model`:

* `sigma.pickle`, `epsilon.pickle`, `nu.pickle`, `mu.pickle`, `charge.pickle` - the original Mpipi parameters (Joseph et al., 2021); `version='Mpipi_original'`. These cover the 20 amino acids only, so this version cannot be used for protein:RNA calculations.
* `sigma_ggv1.pickle`, `epsilon_ggv1.pickle`, `nu_ggv1.pickle`, `mu_ggv1.pickle`, `charge_ggv1.pickle` - the Mpipi-GG parameters (Lotthammer et al., 2024). This is the default (`version='Mpipi_GGv1'`).
* `old_*_ggv1.pickle` - a superseded Mpipi-GG parameter set retained for reproducibility; `version='OLD_Mpipi_GGv1'`. It differs from the current set in the aliphatic-context epsilons, the P-U sigma, and swapped mu/nu values for every U pair.

`build_MpipiGG_v1.ipynb` is the notebook that generated the Mpipi-GG pickles from the original parameters, using the helper functions in `mPiPi_GGv1_modification_fxns.py`. `defined_RNA_params.py` holds the poly-U (rU) parameters that make protein:RNA calculations possible with the Mpipi frontend.

## CALVADOS (`calvados/`)

* `calvados_residues_dict.pickle` - per-residue CALVADOS parameters as a dictionary keyed by one-letter code (three-letter code, molecular weight, bead diameter in nm, charge, and a stickiness value per CALVADOS version); this is the file `finches.forcefields.calvados.CALVADOS_model` reads. It replaced an older pandas-based table, `calvados_residues.pickle`, which has been removed.

## Reference data

* `fingerprints.py` - two reference panels of dipeptide-repeat sequences, `mpipi_fingerprints` and `calvados_fingerprints`, each mapping a dipeptide to a 20-residue repeat of it. Nothing in finches reads them; they are fixed reference sequences to score other sequences against.
* `forcefield_dependencies.py` - `get_null_interaction_baseline()`, which recomputes the null interaction baseline (the value that gives poly(GS) an epsilon of zero) for any forcefield. The charge prefactor cannot be derived automatically and is calibrated in `finches/forcefields/calibration/prefactor_calibration.ipynb`.
* `reference_sequence_info.py` - `DAS_KAPPA_RG_MPIPI`, the 30 Das-Pappu E/K sequences as `(sequence, radius_of_gyration, kappa)` tuples, with radii of gyration in Angstrom from Mpipi-GG simulations, used when calibrating the charge prefactor.
