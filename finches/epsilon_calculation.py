"""
The InteractionMatrixConstructor, which turns a forcefield into FINCHES matrices
and epsilon values.

An InteractionMatrixConstructor wraps a forcefield model (e.g. Mpipi_model or
CALVADOS_model), tabulates the residue-residue interaction parameter for every
pair of residue types, and uses that table to build the raw and weighted
interaction matrices between two sequences, the epsilon value and vectors, and
sliding-window epsilon matrices (intermaps). The epsilon arithmetic itself lives
in finches.epsilon_stateless.

By: Garrett M. Ginell & Alex S. Holehouse
"""

import warnings

import numpy as np

from finches import epsilon_stateless, parsing_aminoacid_sequences
from finches.data import forcefield_dependencies
from finches.utils import matrix_manipulation


def _get_calibrated_config(parameters, key):
    """
    Read a calibration constant out of a forcefield's CONFIGS dictionary.

    Forcefield versions that have not yet been calibrated store their constants
    as NaN (see, for example, CALVADOS1). A NaN passes an ``in CONFIGS``
    membership test but then propagates silently through every downstream
    calculation, so NaN is treated here as equivalent to the key being absent.

    Parameters
    ----------
    parameters : forcefield object
        Forcefield object which may or may not define a CONFIGS dictionary.

    key : str
        Name of the calibration constant to read.

    Returns
    -------
    float or None
        The calibrated value, or None if it is missing or uncalibrated (NaN).

    """
    if not hasattr(parameters, "CONFIGS"):
        return None

    value = parameters.CONFIGS.get(key)

    if value is None or np.isnan(value):
        return None

    return value


# -------------------------------------------------------------------------------------------------
class InteractionMatrixConstructor:
    """
    Build FINCHES interaction matrices and epsilon values from a forcefield.

    On construction the forcefield's interaction parameter is computed for every
    pair of residue types and stored in a lookup table, and the two calibrated
    constants are read from the forcefield's CONFIGS (unless passed explicitly):
    the charge prefactor used by the charge weighting and the null interaction
    baseline that defines zero on the epsilon scale.

    Users normally get one of these from a frontend (Mpipi_frontend().IMC_object
    or CALVADOS_frontend().IMC_object) rather than building it directly.

    Parameters
    ----------
    parameters : forcefield object
        A forcefield model (e.g. finches.forcefields.mpipi.Mpipi_model). It must
        define ALL_RESIDUES_TYPES (a list of lists of residue codes, one list per
        residue group, e.g. protein and RNA) and a
        compute_interaction_parameter(residue_1, residue_2) method whose first
        return value is the interaction parameter.

    sequence_converter : callable or None, optional
        Function applied to every sequence before its matrix is built, to map a
        sequence onto the forcefield's residue codes. It must not change the
        length of the sequence. Default None uses the identity (no conversion).

    charge_prefactor : float, optional
        Scaling factor for the charge weighting, between 0 and 1. Default None
        reads it from parameters.CONFIGS.

    null_interaction_baseline : float, optional
        Baseline that divides attractive from repulsive interactions. Default
        None reads it from parameters.CONFIGS.

    compute_forcefield_dependencies : bool, optional
        If True and the forcefield has no calibrated null_interaction_baseline,
        calibrate one against poly(GS) with
        finches.data.forcefield_dependencies.get_null_interaction_baseline().
        Default False.

    Attributes
    ----------
    parameters : forcefield object
        The forcefield model.

    lookup : dict
        Nested dictionary, lookup[r1][r2] = interaction parameter between residue
        types r1 and r2.

    charge_prefactor : float
        The charge prefactor in use.

    null_interaction_baseline : float
        The null interaction baseline in use.

    sequence_converter : callable
        The sequence converter in use.

    valid_residue_groups : list of list of str
        The forcefield's residue groups (parameters.ALL_RESIDUES_TYPES).

    Raises
    ------
    AttributeError
        If parameters has no ALL_RESIDUES_TYPES attribute.

    ValueError
        If the null interaction baseline or charge prefactor is neither passed
        nor calibrated for the forcefield (and, for the baseline,
        compute_forcefield_dependencies is False).

    """

    def __init__(
        self,
        parameters,
        sequence_converter=None,
        charge_prefactor=None,
        null_interaction_baseline=None,
        compute_forcefield_dependencies=False,
    ):
        """
        Build the lookup table and resolve the calibrated constants.

        See the class docstring for the parameters.
        """

        # Validate parameters object
        if not hasattr(parameters, "ALL_RESIDUES_TYPES"):
            raise AttributeError(
                "Parameters object must have ALL_RESIDUES_TYPES attribute"
            )

        # Initialize core variables. sequence_converter used to default to False,
        # so any falsy value means "no conversion"
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
            config_baseline = _get_calibrated_config(
                self.parameters, "null_interaction_baseline"
            )
            if config_baseline is not None:
                self.null_interaction_baseline = config_baseline
            elif compute_forcefield_dependencies:
                print(
                    f"Recomputing null_interaction_baseline for {self.parameters.version}..."
                )
                self.null_interaction_baseline = (
                    forcefield_dependencies.get_null_interaction_baseline(self)
                )
            else:
                raise ValueError(
                    f"null_interaction_baseline has not been calibrated for "
                    f"{self.parameters.version}. Pass one explicitly, or construct with "
                    f"compute_forcefield_dependencies=True to calibrate it."
                )

        if self.charge_prefactor is None:
            config_prefactor = _get_calibrated_config(
                self.parameters, "charge_prefactor"
            )
            if config_prefactor is None:
                raise ValueError(
                    f"charge_prefactor has not been calibrated for "
                    f"{self.parameters.version}. Pass one explicitly, or define it in "
                    f"the forcefield CONFIGS."
                )
            self.charge_prefactor = config_prefactor

    def _update_lookup_dict(self, unknown_set_to_zero=False):
        """
        Recompute the residue-residue interaction lookup table.

        Every pair of residue types across all residue groups is passed to the
        forcefield's compute_interaction_parameter(). Call this after changing
        the forcefield's conditions (salt, pH, ...) so the table reflects them.

        Parameters
        ----------
        unknown_set_to_zero : bool, optional
            If True, a residue pair the forcefield has no parameters for is set
            to zero with a warning instead of raising. Default False.

        Raises
        ------
        ValueError
            If the forcefield has no parameters for a residue pair and
            unknown_set_to_zero is False.
        """
        self.lookup = {}
        valid_aa = sorted(
            {res for sublist in self.valid_residue_groups for res in sublist}
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
                        warnings.warn(
                            f"Unknown residue pair {r1}-{r2}, setting its interaction "
                            "to zero."
                        )
                        self.lookup[r1][r2] = 0.0
                    else:
                        raise ValueError(
                            f"The forcefield has no parameters for the residue pair "
                            f"{r1}-{r2} ({e!r})."
                        ) from e

        # build a vectorized NumPy lookup table so the pairwise matrix can be
        # constructed by integer array indexing instead of per-element dict lookups
        self._build_lookup_matrix()

    def _build_lookup_matrix(self):
        """
        Build a NumPy table equivalent to self.lookup for fast matrix construction.

        Sets three attributes:

        - ``self._lookup_matrix`` : (n_res, n_res) float array where
          ``[i, j]`` is the interaction parameter for the residues with codes
          ``i`` and ``j``.
        - ``self._residue_codes`` : dict mapping residue character -> code.
        - ``self._ascii_to_code`` : length-256 int array mapping a residue's
          byte value -> code (-1 for bytes that are not valid residues), so a
          sequence can be turned into codes with a single array index.

        If any residue key is not a single character with an ordinal < 256 the
        three attributes are set to None and matrices are built from the
        dictionary instead.

        Returns
        -------
        None
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
        Swap in a forcefield model and rebuild the lookup table.

        The charge prefactor, sequence converter and null interaction baseline
        are not changed.

        Parameters
        ----------
        new_parameters : forcefield object
            The new forcefield model (see the class docstring for what it must
            provide).

        Returns
        -------
        None
        """
        self.parameters = new_parameters
        self.valid_residue_groups = new_parameters.ALL_RESIDUES_TYPES
        self._update_lookup_dict()

    def _check_sequence(self, sequence):
        """
        Check that a sequence can be scored by this forcefield.

        Parameters
        ----------
        sequence : str
            The (converted) sequence to check.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the sequence is empty, contains a residue the forcefield does not
            know, or mixes residues from more than one residue group (for
            example protein and RNA residues).
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
        Return a sequence passed through the sequence converter and validated.

        Parameters
        ----------
        sequence : str
            The sequence to convert.

        Returns
        -------
        str
            The converted sequence.

        Raises
        ------
        ValueError
            If the converted sequence is empty, contains an unknown residue, or
            mixes residue groups.
        """
        outseq = self.sequence_converter(sequence)
        self._check_sequence(outseq)
        return outseq

    # Matrix calculation functions

    def calculate_pairwise_homotypic_matrix(
        self, sequence, convert_to_custom=True, use_cython=True
    ):
        """
        Calculate the raw interaction matrix of a sequence with itself.

        Identical to calculate_pairwise_heterotypic_matrix(sequence, sequence).

        Parameters
        ----------
        sequence : str
            The sequence.

        convert_to_custom : bool, optional
            Whether to pass the sequence through the sequence converter first.
            Default True.

        use_cython : bool, optional
            See calculate_pairwise_heterotypic_matrix(). Default True.

        Returns
        -------
        np.ndarray
            Symmetric matrix of shape (len(sequence), len(sequence)).
        """
        return self.calculate_pairwise_heterotypic_matrix(
            sequence,
            sequence,
            convert_to_custom=convert_to_custom,
            use_cython=use_cython,
        )

    def calculate_pairwise_heterotypic_matrix(
        self, sequence1, sequence2, convert_to_custom=True, use_cython=True
    ):
        """
        Calculate the raw interaction matrix between two sequences.

        Element (i, j) is the forcefield interaction parameter between residue i
        of sequence1 and residue j of sequence2, where more negative values are
        more attractive. Whether an element counts as attractive or repulsive in
        the epsilon calculation is decided relative to the null interaction
        baseline, not relative to zero. No context weighting is applied (see
        calculate_weighted_pairwise_matrix()).

        Parameters
        ----------
        sequence1 : str
            First sequence (the rows of the matrix).

        sequence2 : str
            Second sequence (the columns of the matrix).

        convert_to_custom : bool, optional
            Whether to pass both sequences through the sequence converter
            first. Default True.

        use_cython : bool, optional
            Only used when the forcefield's residue codes cannot use the
            vectorized NumPy table (i.e. a residue code is not a single latin-1
            character): True builds the matrix from the lookup dictionary with
            the Cython dict2matrix(), False with pure Python. The result is the
            same either way. Default True.

        Returns
        -------
        np.ndarray
            Matrix of shape (len(sequence1), len(sequence2)).

        Raises
        ------
        ValueError
            If either (converted) sequence is empty, contains an unknown
            residue, or mixes residue groups (e.g. protein and RNA residues).
        """
        if convert_to_custom:
            sequence1 = self.sequence_converter(sequence1)
            sequence2 = self.sequence_converter(sequence2)

        self._check_sequence(sequence1)
        self._check_sequence(sequence2)

        # fast path: vectorized NumPy lookup-table indexing, identical to the
        # dict-based construction below but ~8x faster. _check_sequence() has
        # already rejected unknown residues, so every code is valid; the check
        # below is only a safety net
        if self._lookup_matrix is not None:
            codes1 = self._ascii_to_code[
                np.frombuffer(
                    sequence1.encode("latin-1", errors="replace"), dtype=np.uint8
                )
            ]
            codes2 = self._ascii_to_code[
                np.frombuffer(
                    sequence2.encode("latin-1", errors="replace"), dtype=np.uint8
                )
            ]
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
        Calculate the context-weighted interaction matrix between two sequences.

        The raw matrix from calculate_pairwise_heterotypic_matrix() is corrected
        for local sequence context:

        * charge weighting: M_ij -> M_ij * (1 - prefactor * w_ij), where w_ij is
          the charge weight from
          finches.parsing_aminoacid_sequences.get_charge_weighted_mask(), which
          reduces like-charge repulsion between clusters of like charges;
        * aliphatic weighting: M_ij -> M_ij * a_ij, where a_ij (1, 1.5 or 3) is
          from get_aliphatic_weighted_mask() and up-weights contacts between
          aliphatic clusters.

        The context is read from the amino acid sequences as passed in, before
        any sequence conversion, because it is a property of the amino acid
        sequence rather than of the forcefield's residue codes.

        Parameters
        ----------
        sequence1 : str
            First sequence (the rows of the matrix).

        sequence2 : str
            Second sequence (the columns of the matrix).

        convert_to_custom : bool, optional
            Whether to pass both sequences through the sequence converter
            before building the raw matrix. Default True.

        charge_prefactor : float, optional
            Charge prefactor for the charge weighting. Default None uses
            self.charge_prefactor; an explicit 0 is honoured.

        use_charge_weighting : bool, optional
            Whether to apply the charge weighting. Default True.

        use_aliphatic_weighting : bool, optional
            Whether to apply the aliphatic weighting. Default True.

        use_cython : bool, optional
            Passed to calculate_pairwise_heterotypic_matrix(). Default True.

        Returns
        -------
        np.ndarray
            Matrix of shape (len(sequence1), len(sequence2)).

        Raises
        ------
        ValueError
            If a sequence is invalid (see
            calculate_pairwise_heterotypic_matrix()), if the sequence converter
            changed a sequence's length, or if charge weighting is requested
            without a charge prefactor.
        """
        matrix = self.calculate_pairwise_heterotypic_matrix(
            sequence1,
            sequence2,
            convert_to_custom=convert_to_custom,
            use_cython=use_cython,
        )

        # the context masks are built on the sequences as passed in, so they only
        # line up with the matrix if the converter preserved the lengths
        if (use_charge_weighting or use_aliphatic_weighting) and matrix.shape != (
            len(sequence1),
            len(sequence2),
        ):
            raise ValueError(
                "The sequence converter changed the length of a sequence "
                f"({len(sequence1)}x{len(sequence2)} -> {matrix.shape[0]}x"
                f"{matrix.shape[1]}), so the context weighting cannot be applied."
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
        Calculate the attractive and repulsive epsilon vectors between two sequences.

        A wrapper around
        finches.epsilon_stateless.get_sequence_epsilon_vectors() that uses this
        constructor's charge prefactor and null interaction baseline.

        Parameters
        ----------
        sequence1 : str
            The sequence the vectors are reported for.

        sequence2 : str
            The sequence it is compared against.

        use_charge_weighting : bool, optional
            Whether to apply the charge weighting. Default True.

        use_aliphatic_weighting : bool, optional
            Whether to apply the aliphatic weighting. Default True.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (attractive_vector, repulsive_vector), each of length
            len(sequence1). Their sum over both vectors is the epsilon value.
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
        Calculate the epsilon value between two sequences.

        A wrapper around finches.epsilon_stateless.get_sequence_epsilon_value()
        that uses this constructor's charge prefactor and null interaction
        baseline. Epsilon is the sum over the residues of sequence1 of the mean
        over the residues of sequence2 of M_ij - 2b; negative is attractive,
        positive repulsive, and poly(GS) against itself is zero.

        Parameters
        ----------
        sequence1 : str
            The first sequence (epsilon scales with its length).

        sequence2 : str
            The second sequence (epsilon is averaged over it).

        use_charge_weighting : bool, optional
            Whether to apply the charge weighting. Default True.

        use_aliphatic_weighting : bool, optional
            Whether to apply the aliphatic weighting. Default True.

        Returns
        -------
        float
            The epsilon value.
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
        Calculate the sliding-window epsilon matrix (intermap) between two sequences.

        The weighted interaction matrix is built once for the full sequences, so
        every window keeps its true sequence context, and the epsilon of every
        window_size x window_size sub-matrix is then computed.

        Parameters
        ----------
        sequence1 : str
            First sequence (axis 0 of the output).

        sequence2 : str
            Second sequence (axis 1 of the output).

        window_size : int, optional
            Window size. Must be positive and no longer than either sequence; an
            even value is rounded up to the next odd one with a warning.
            Default 31.

        use_charge_weighting : bool, optional
            Whether to apply the charge weighting. Default True.

        use_aliphatic_weighting : bool, optional
            Whether to apply the aliphatic weighting. Default True.

        use_cython : bool, optional
            If True use the summed-area-table implementation in
            finches.utils.matrix_manipulation.matrix_scan(); if False use a
            (much slower) pure-Python loop. The results are the same. Default
            True.

        Returns
        -------
        tuple
            (epsilon_matrix, seq1_indices, seq2_indices). epsilon_matrix has
            shape (len(sequence1) - window_size + 1, len(sequence2) -
            window_size + 1), axis 0 indexing sequence1 and axis 1 sequence2.
            seq1_indices and seq2_indices are the 1-based residue positions of
            the window centres along each axis.

        Raises
        ------
        ValueError
            If window_size is less than 1 or longer than either sequence, or if
            a sequence is invalid.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {window_size}")

        # Ensure odd window size
        if window_size % 2 == 0:
            warnings.warn(f"window size is even, rounding up to {window_size + 1}")
            window_size = window_size + 1

        # Calculate weighted pairwise matrix
        w_matrix = self.calculate_weighted_pairwise_matrix(
            sequence1,
            sequence2,
            use_charge_weighting=use_charge_weighting,
            use_aliphatic_weighting=use_aliphatic_weighting,
        )

        # check the window fits before dispatching so that the Cython and Python
        # paths raise the same error
        l1, l2 = w_matrix.shape
        if l1 < window_size or l2 < window_size:
            raise ValueError("Window size larger than matrix size")

        if use_cython:
            return matrix_manipulation.matrix_scan(
                w_matrix, window_size, self.null_interaction_baseline
            )

        # Python fallback: the epsilon of every window sub-matrix
        everything = np.empty((l1 - window_size + 1, l2 - window_size + 1))
        for i in range(everything.shape[0]):
            for j in range(everything.shape[1]):
                attractive, repulsive = epsilon_stateless.get_matrix_epsilon_vectors(
                    w_matrix[i : i + window_size, j : j + window_size],
                    self.null_interaction_baseline,
                )
                everything[i, j] = np.sum(attractive) + np.sum(repulsive)

        # 1-based residue number of each window centre, axis 0 = sequence1 and
        # axis 1 = sequence2, matching the Cython path
        start = (window_size - 1) // 2 + 1
        seq1_indices = np.arange(start, l1 - start + 2)
        seq2_indices = np.arange(start, l2 - start + 2)

        return (everything, seq1_indices, seq2_indices)
