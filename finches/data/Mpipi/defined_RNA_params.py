"""
Mpipi parameters for the RNA nucleotide U.

Taken from the Mpipi LAMMPS input (lammps_Mpipi_v5_NEWPARAMS.in) on 2023-08-16
and used by mPiPi_GGv1_modification_fxns.ADD_RNA_U_ALL() to add U to the
Mpipi-GG parameter set.

* CHARGE maps U to its charge in units of the elementary charge.
* PARAMS maps each residue to its parameters with U, as
  [partner, epsilon (kJ/mol), sigma (Angstroms), nu, mu]; HEADER names the
  columns.
"""

CHARGE = {"U": -0.75}

HEADER = {"R1": ["R2", "EPSILON", "SIGMA", "NU", "MU"]}

PARAMS = {
    "M": ["U", 0.172282, 7.318975, 1.0, 3.0],
    "G": ["U", 0.200735, 6.432555, 1.0, 3.0],
    "K": ["U", 0.097235, 7.420670, 1.0, 3.0],
    "T": ["U", 0.167879, 7.029531, 1.0, 3.0],
    "R": ["U", 0.394916, 7.504526, 1.0, 3.0],
    "A": ["U", 0.177240, 6.720037, 1.0, 3.0],
    "D": ["U", 0.192048, 6.996761, 1.0, 3.0],
    "E": ["U", 0.195311, 7.173835, 1.0, 3.0],
    "Y": ["U", 0.362093, 7.451817, 1.0, 3.0],
    "V": ["U", 0.155289, 7.218000, 1.0, 3.0],
    "L": ["U", 0.157999, 7.352036, 1.0, 3.0],
    "Q": ["U", 0.252724, 7.223925, 1.0, 3.0],
    "W": ["U", 0.427648, 7.618274, 1.0, 3.0],
    "F": ["U", 0.348321, 7.399775, 1.0, 3.0],
    "S": ["U", 0.183300, 6.791335, 1.0, 3.0],
    "H": ["U", 0.166108, 7.253891, 1.0, 3.0],
    "N": ["U", 0.249424, 7.046675, 1.0, 3.0],
    "P": ["U", 0.191838, 6.988078, 1.0, 3.0],
    "C": ["U", 0.187156, 6.947181, 1.0, 3.0],
    "I": ["U", 0.152697, 7.545840, 1.0, 3.0],
    "U": ["U", 0.305000, 8.170000, 1.0, 3.0],
}
