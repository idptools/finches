"""
Class to build Interation Matrix from Mpipi forcefield By


values : Garrett M. Ginell & Alex S. Holehouse
2023-08-06
"""

import numpy as np

from finches import epsilon_stateless, parsing_aminoacid_sequences
from finches.data import forcefield_dependencies
from finches.utils import matrix_manipulation


# -------------------------------------------------------------------------------------------------
class InteractionMatrixConstructor:
    def __init__(
        self,
        parameters,
        sequence_converter=False,
        charge_prefactor=None,
        null_interaction_baseline=None,
        compute_forcefield_dependencies=False,
    ):
        """
        Constructor for calculating inter-residue interactions using forcefield models.
        Provides standardized interface for different biophysical models.

        Parameters
        -----------
        parameters : forcefield object
            Must have ALL_RESIDUES_TYPES and compute_interaction_parameter() method
        sequence_converter : function, optional
            Function to convert sequences to valid residue types
        charge_prefactor : float, optional
            Scaling factor for charge weighting (0-1)
        null_interaction_baseline : float, optional
            Threshold for attractive vs repulsive interactions
        compute_forcefield_dependencies : bool
            Whether to recompute missing forcefield parameters
        """

        # Validate parameters object
        if not hasattr(parameters, "ALL_RESIDUES_TYPES"):
            raise AttributeError(
                "Parameters object must have ALL_RESIDUES_TYPES attribute"
            )

        # Initialize core variables
        self.parameters = None
        self.valid_residue_groups = parameters.ALL_RESIDUES_TYPES
        self.sequence_converter = sequence_converter or (lambda a: a)
        self.charge_prefactor = charge_prefactor
        self.null_interaction_baseline = null_interaction_baseline
        self.lookup = {}
        # NumPy lookup table for fast pairwise-matrix construction; populated by
        # _build_lookup_matrix() when the lookup dict is built below
        self._lookup_matrix = None
        self._residue_codes = None
        self._ascii_to_code = None

        # Set up parameters and lookup table
        self._update_parameters(parameters)

        # Set defaults from forcefield configs if not provided
        if self.null_interaction_baseline is None:
            if (
                hasattr(self.parameters, "CONFIGS")
                and "null_interaction_baseline" in self.parameters.CONFIGS
            ):
                self.null_interaction_baseline = self.parameters.CONFIGS[
                    "null_interaction_baseline"
                ]
            elif compute_forcefield_dependencies:
                print(
                    f"Recomputing null_interaction_baseline for {self.parameters.version}..."
                )
                self.null_interaction_baseline = (
                    forcefield_dependencies.get_null_interaction_baseline(self)
                )
            else:
                print(
                    f"WARNING: null_interaction_baseline not found for {parameters.version}"
                )

        if self.charge_prefactor is None:
            if (
                hasattr(self.parameters, "CONFIGS")
                and "charge_prefactor" in self.parameters.CONFIGS
            ):
                self.charge_prefactor = self.parameters.CONFIGS["charge_prefactor"]
            else:
                raise ValueError(
                    "charge_prefactor must be provided or defined in forcefield CONFIGS"
                )

    def _update_lookup_dict(self, unknown_set_to_zero=False):
        """
        Recalculate inter-residue interaction lookup table.

        Parameters
        ----------
        unknown_set_to_zero : bool
            Set unknown interactions to zero instead of raising error
        """
        self.lookup = {}
        valid_aa = list(
            set([res for sublist in self.valid_residue_groups for res in sublist])
        )

        for r1 in valid_aa:
            self.lookup[r1] = {}
            for r2 in valid_aa:
                try:
                    self.lookup[r1][r2] = self.parameters.compute_interaction_parameter(
                        r1, r2
                    )[0]
                except KeyError as e:
                    if unknown_set_to_zero:
                        print(
                            f"WARNING: Unknown residue pair {r1}-{r2}, setting to zero."
                        )
                        self.lookup[r1][r2] = 0.0
                    else:
                        raise Exception(f"ERROR: {e} for {r1} and {r2}.")

        # build a vectorized NumPy lookup table so the pairwise matrix can be
        # constructed by integer array indexing instead of per-element dict lookups
        self._build_lookup_matrix()

    def _build_lookup_matrix(self):
        """
        Build a NumPy lookup table equivalent to ``self.lookup`` for fast,
        vectorized pairwise-matrix construction.

        Sets three attributes:

        - ``self._lookup_matrix`` : (n_res, n_res) float array where
          ``[i, j]`` is the interaction parameter for the residues with codes
          ``i`` and ``j``.
        - ``self._residue_codes`` : dict mapping residue character -> code.
        - ``self._ascii_to_code`` : length-256 int array mapping a residue's
          byte value -> code (-1 for bytes that are not valid residues), so a
          sequence can be turned into codes with a single array index.

        If any residue key is not a single character with an ordinal < 256 the
        table is set to ``None`` and the dict-based path is used instead.
        """
        residues = sorted(self.lookup.keys())

        # only single-character residues with ordinals < 256 can use the fast
        # ascii/latin-1 byte-indexed path; otherwise disable it and fall back
        if not all(len(r) == 1 and ord(r) < 256 for r in residues):
            self._lookup_matrix = None
            self._residue_codes = None
            self._ascii_to_code = None
            return

        self._residue_codes = {r: i for i, r in enumerate(residues)}

        n = len(residues)
        table = np.empty((n, n), dtype=float)
        for r1, i in self._residue_codes.items():
            for r2, j in self._residue_codes.items():
                table[i, j] = self.lookup[r1][r2]
        self._lookup_matrix = table

        ascii_to_code = np.full(256, -1, dtype=np.intp)
        for r, i in self._residue_codes.items():
            ascii_to_code[ord(r)] = i
        self._ascii_to_code = ascii_to_code

    def _update_parameters(self, new_parameters):
        """
        Update forcefield parameters and rebuild lookup table.
        Note: Does not update charge_prefactor, sequence_converter, or null_interaction_baseline.
        """
        self.parameters = new_parameters
        self.valid_residue_groups = new_parameters.ALL_RESIDUES_TYPES
        self._update_lookup_dict()

    def _check_sequence(self, sequence):
        """
        Validate that sequence contains residues from only one residue group.
        """
        if not sequence:
            raise ValueError("Empty sequence provided")

        unique_residues = set(sequence)
        matching_groups = 0
        total_matches = 0

        for residue_group in self.valid_residue_groups:
            matches = unique_residues.intersection(residue_group)
            if matches:
                matching_groups += 1
                total_matches += len(matches)

        if matching_groups > 1:
            raise ValueError(
                f"Sequence contains residues from multiple groups: {sequence}"
            )
        elif total_matches < len(unique_residues):
            raise ValueError(f"Unknown residue found in sequence: {sequence}")

    def get_converted_sequence(self, sequence):
        """
        Return sequence passed through the sequence converter.
        """
        outseq = self.sequence_converter(sequence)
        self._check_sequence(outseq)
        return outseq

    # Matrix calculation functions

    def calculate_pairwise_homotypic_matrix(self, sequence, convert_to_custom=True):
        """
        Calculate pairwise interaction matrix for a single sequence.
        """
        return self.calculate_pairwise_heterotypic_matrix(
            sequence, sequence, convert_to_custom=convert_to_custom
        )

    def calculate_pairwise_heterotypic_matrix(
        self, sequence1, sequence2, convert_to_custom=True, use_cython=True
    ):
        """
        Calculate pairwise interaction matrix between two sequences.

        Returns (len(s1) x len(s2)) matrix where negative values are attractive
        and positive values are repulsive.
        """
        if convert_to_custom:
            sequence1 = self.sequence_converter(sequence1)
            sequence2 = self.sequence_converter(sequence2)
        else:
            self._check_sequence(sequence1)
            self._check_sequence(sequence2)

        # fast path: vectorized NumPy lookup-table indexing. This is identical to
        # the dict-based construction below but ~8x faster, and is the default
        # whenever the table is available (see _build_lookup_matrix).
        if self._lookup_matrix is not None:
            codes1 = self._ascii_to_code[
                np.frombuffer(sequence1.encode("latin-1"), dtype=np.uint8)
            ]
            codes2 = self._ascii_to_code[
                np.frombuffer(sequence2.encode("latin-1"), dtype=np.uint8)
            ]
            # if every residue is known, index the table directly; otherwise fall
            # through to the dict path so an unknown residue raises as before
            if not (codes1 < 0).any() and not (codes2 < 0).any():
                return self._lookup_matrix[codes1[:, None], codes2[None, :]]

        if use_cython:
            return matrix_manipulation.dict2matrix(sequence1, sequence2, self.lookup)
        else:
            return np.array(
                [[self.lookup[r1][r2] for r2 in sequence2] for r1 in sequence1]
            )

    def calculate_weighted_pairwise_matrix(
        self,
        sequence1,
        sequence2,
        convert_to_custom=True,
        charge_prefactor=None,
        use_charge_weighting=True,
        use_aliphatic_weighting=True,
        use_cython=True,
    ):
        """
        Calculate weighted pairwise matrix with charge and aliphatic weighting.
        """
        matrix = self.calculate_pairwise_heterotypic_matrix(
            sequence1,
            sequence2,
            convert_to_custom=convert_to_custom,
            use_cython=use_cython,
        )

        w_matrix = matrix

        if use_charge_weighting:
            # an explicitly-passed prefactor of 0 is valid and must not fall
            # through to the instance default
            prefactor = (
                self.charge_prefactor if charge_prefactor is None else charge_prefactor
            )
            if prefactor is None:
                raise ValueError(
                    "charge_prefactor must be defined for charge weighting"
                )

            _, repulsive_mask = parsing_aminoacid_sequences.get_charge_weighted_mask(
                sequence1, sequence2
            )
            w_matrix = matrix - (matrix * repulsive_mask * prefactor)

        if use_aliphatic_weighting:
            w_ali_mask = parsing_aminoacid_sequences.get_aliphatic_weighted_mask(
                sequence1, sequence2
            )
            w_matrix = w_matrix * w_ali_mask

        return w_matrix

    # Epsilon calculation functions

    def calculate_epsilon_vectors(
        self,
        sequence1,
        sequence2,
        use_charge_weighting=True,
        use_aliphatic_weighting=True,
    ):
        """
        Calculate attractive and repulsive epsilon vectors for two sequences.

        Returns
        -------
        tuple
            (attractive_vector, repulsive_vector)
        """
        return epsilon_stateless.get_sequence_epsilon_vectors(
            sequence1,
            sequence2,
            self,
            use_charge_weighting=use_charge_weighting,
            use_aliphatic_weighting=use_aliphatic_weighting,
        )

    def calculate_epsilon_value(
        self,
        sequence1,
        sequence2,
        use_charge_weighting=True,
        use_aliphatic_weighting=True,
    ):
        """
        Calculate overall epsilon value for two sequences.

        Returns
        -------
        float
            Average sequence-sequence interaction value
        """
        return epsilon_stateless.get_sequence_epsilon_value(
            sequence1,
            sequence2,
            self,
            use_charge_weighting=use_charge_weighting,
            use_aliphatic_weighting=use_aliphatic_weighting,
        )

    def calculate_sliding_epsilon(
        self,
        sequence1,
        sequence2,
        window_size=31,
        use_charge_weighting=True,
        use_aliphatic_weighting=True,
        use_cython=True,
    ):
        """
        Calculate sliding window epsilon values between two sequences.

        Returns
        -------
        tuple
            (epsilon_matrix, seq1_indices, seq2_indices) where epsilon_matrix has
            shape (len(seq1_indices), len(seq2_indices)); i.e. axis 0 indexes
            sequence1 and axis 1 indexes sequence2. Indices are 1-based protein
            positions corresponding to each window centre.
        """

        def __matrix2eps(in_matrix):
            """Calculate epsilon value for a matrix."""
            attractive_matrix, repulsive_matrix = (
                epsilon_stateless.get_attractive_repulsive_matrices(
                    in_matrix, self.null_interaction_baseline
                )
            )
            attractive_matrix = attractive_matrix - self.null_interaction_baseline
            repulsive_matrix = repulsive_matrix - self.null_interaction_baseline
            return np.sum(np.mean(attractive_matrix, axis=1)) + np.sum(
                np.mean(repulsive_matrix, axis=1)
            )

        # Ensure odd window size
        if window_size % 2 == 0:
            print(f"Warning: window size is even, rounding up to {window_size + 1}")
            window_size = window_size + 1

        # Calculate weighted pairwise matrix
        w_matrix = self.calculate_weighted_pairwise_matrix(
            sequence1,
            sequence2,
            use_charge_weighting=use_charge_weighting,
            use_aliphatic_weighting=use_aliphatic_weighting,
        )

        # Use cython implementation if available
        if use_cython:
            return matrix_manipulation.matrix_scan(
                w_matrix, window_size, self.null_interaction_baseline
            )

        # Python fallback implementation
        l1, l2 = w_matrix.shape
        if l1 < window_size or l2 < window_size:
            raise ValueError("Window size larger than matrix size")

        # Calculate sliding epsilon for all windows
        everything = []
        for i in range((l1 - window_size) + 1):
            row = []
            for j in range((l2 - window_size) + 1):
                row.append(
                    __matrix2eps(w_matrix[i : i + window_size, j : j + window_size])
                )
            everything.append(row)

        everything = np.array(everything)

        # Calculate indices (1-based for protein numbering). everything has shape
        # (l1 - window + 1, l2 - window + 1), so axis 0 maps to sequence1 and axis 1
        # to sequence2 - return the indices in that order to match the Cython path.
        start = (window_size - 1) // 2 + 1
        seq1_indices = np.arange(start, l1 - start + 2)
        seq2_indices = np.arange(start, l2 - start + 2)

        return (everything, seq1_indices, seq2_indices)
