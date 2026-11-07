# Calibration notebooks

Notebooks used to calibrate the CALVADOS2 constants in FINCHES against Mpipi-GG. They are kept as a record of how the values in `finches/forcefields/calvados.py` (`CALVADOS_CONFIGS`) were chosen, not as part of the package.

* `prefactor_calibration.ipynb` - how the charge prefactor was chosen (0.2 for Mpipi-GG, 0.7 for CALVADOS2), using the Das-Pappu E/K sequences.
* `calvados_param_check.ipynb` - a quick look at the CALVADOS residue parameters as shipped in `finches/data/calvados/calvados_residues_dict.pickle`, including the pH-dependent histidine charge and the pairwise charge products.
