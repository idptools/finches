"""
Folded-domain utilities: the original patch mode for IDR:folded domain intermaps.

This module holds the FoldedDomain class, which reads a folded domain from a PDB
file, classifies its residues as surface or buried from their solvent accessible
surface area, groups surface residues into straight-line patches, and scores those
patches against an IDR with FINCHES. It is the original ("patch mode") route to
IDR:folded domain intermaps and is kept so that earlier results can be reproduced;
for new work use finches.utils.structure_surface.StructureSurface, which decomposes
the structure into folded domains and IDRs, works over a contiguous-surface net and
models the contact footprint of an IDR window.

The module also provides the residue tables (MAX_SASA_DATA, THREE_TO_ONE,
NONSTANDARD_TO_ONE and one_letter_code()) that StructureSurface shares, and
extract_and_write_domains() for excising a domain from a larger structure.
"""

import warnings

import mdtraj as md
import numpy as np

from scipy.spatial import distance
import networkx as nx

from finches.parsing_aminoacid_sequences import (
    ALIPHATIC_RESIDUES,
    NEGATIVE_RESIDUES,
    POSITIVE_RESIDUES,
)


# Maximum solvent accessible surface area (Angstrom^2) of each residue type as
# [side chain, backbone], calculated from all-atom excluded-volume simulations of
# GXG tripeptides (i.e. with all attractive non-bonded interactions turned off). A
# residue is called surface when its SASA exceeds a fraction of these maxima.
MAX_SASA_DATA = {
    "A": [7.581871795654296875e01, 7.607605743408203125e01],
    "C": [1.154064483642578125e02, 6.787722015380859375e01],
    "D": [1.302558288574218750e02, 7.182710266113281250e01],
    "E": [1.617985687255859375e02, 6.805746459960937500e01],
    "F": [2.093871002197265625e02, 6.598278808593750000e01],
    "G": [0.000000000000000000e00, 1.149752731323242188e02],
    "H": [1.808149414062500000e02, 6.750666809082031250e01],
    "I": [1.727196502685546875e02, 6.034464645385742188e01],
    "K": [2.058575897216796875e02, 6.871156311035156250e01],
    "L": [1.720360412597656250e02, 6.451246643066406250e01],
    "M": [1.847660064697265625e02, 6.778076934814453125e01],
    "N": [1.427441253662109375e02, 6.680493164062500000e01],
    "P": [1.342914733886718750e02, 5.583909606933593750e01],
    "Q": [1.733262939453125000e02, 6.660184478759765625e01],
    "R": [2.364875640869140625e02, 6.673487854003906250e01],
    "S": [9.587133026123046875e01, 7.287202453613281250e01],
    "T": [1.309214324951171875e02, 6.421310424804687500e01],
    "V": [1.431178131103515625e02, 6.172962188720703125e01],
    "W": [2.545694122314453125e02, 6.430991363525390625e01],
    "Y": [2.225183105468750000e02, 7.186695098876953125e01],
}

# amino acid conversion
THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

# One-letter codes for residue names that turn up in real structure files but are not
# the canonical twenty: protonation and disulfide variants written by MD engines, and
# selenomethionine. These take precedence over mdtraj's own residue-name table, which
# for example maps GLH (protonated glutamate) to glutamine.
NONSTANDARD_TO_ONE = {
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",
    "CYX": "C",
    "CYM": "C",
    "ASH": "D",
    "GLH": "E",
    "LYN": "K",
    "ARN": "R",
    "MSE": "M",
}

_STANDARD_LETTERS = frozenset(THREE_TO_ONE.values())


def one_letter_code(residue):
    """
    Return the one-letter amino acid code for an mdtraj residue.

    The residue name is looked up in ``THREE_TO_ONE``, then in ``NONSTANDARD_TO_ONE``
    (protonation/disulfide variants and selenomethionine), and finally in mdtraj's
    own residue-name table, whose answer is accepted only if it is one of the twenty
    standard letters. This is the one place structure-reading code should go to turn
    a residue into a letter, so that an unusual residue name is never silently
    turned into the wrong chemistry.

    Parameters
    ----------
    residue : mdtraj.core.topology.Residue
        The residue to translate.

    Returns
    -------
    str or None
        The one-letter code, or None if the residue has no standard amino acid
        equivalent (terminal caps, ligands, waters, unknown modified residues).
    """
    aa = THREE_TO_ONE.get(residue.name)
    if aa is None:
        aa = NONSTANDARD_TO_ONE.get(residue.name)
    if aa is None:
        code = residue.code
        if code in _STANDARD_LETTERS:
            aa = code
    return aa


# internal notes 2024-10-20
# NB: In earlier versions of finches, FoldedDomain was defined as FoldeDomain (missing the
# ending 'd' of folded. To maintain backwards compatibility, we alias the old name to the
# new correct one at the end of this class.
class FoldedDomain:
    """
    A folded domain read from a PDB file, for the patch mode of IDR:folded domain
    intermaps.

    On construction the structure is loaded, the solvent accessible surface area
    (SASA) of every residue is computed with the Shrake-Rupley algorithm, and each
    residue is classified as surface or buried. The surface neighbour graph and the
    distances between surface residues are built lazily, on first access to any of
    the properties that need them.

    Every protein residue in the file is treated as part of the folded domain. For a
    structure that also contains disordered regions, or for new work in general, use
    finches.utils.structure_surface.StructureSurface instead.

    Parameters
    ----------
    pdbfilename : str
        Path to the PDB file.

    start, end : int, optional
        0-based mdtraj residue indices (inclusive) of a sub-range to keep. Either
        both or neither must be given. This path is not well tested, so it raises
        unless ``ignore_warnings=True``; we recommend excising the domain with
        extract_and_write_domains() and passing the new file instead. Default None.

    probe_radius : float, optional
        Probe radius for the SASA calculation, in Angstrom. Default 1.4.

    residue_overide_mapping : dict, optional
        Maps residue names that have no standard amino acid equivalent (see
        one_letter_code()) to a one-letter code, e.g. ``{'XYZ': 'K'}``. Default None.

    surface_thresh : float, optional
        Fraction of the maximum SASA (MAX_SASA_DATA) above which a residue is called
        surface. Default 0.10.

    sasa_mode : str, optional
        ``'v1'`` (default) compares the residue SASA to the maximum side-chain SASA
        only. Glycine has no side chain, so it counts as surface unless its SASA is
        exactly zero. ``'v2'`` compares it to the maximum side-chain plus backbone
        SASA, which is more stringent.

    ignore_warnings : bool, optional
        If True, allow the untested ``start``/``end`` path. Default False.

    SASA_ONLY : bool, optional
        If True, stop after computing SASA and the sequence. Nothing else is set up,
        so every other method will fail. Default False.

    SASA_override : dict, optional
        Maps 0-based residue indices to SASA values (Angstrom^2) that replace the
        calculated ones. A value above 250 forces a residue to the surface and 0
        forces it to be buried. Default None.

    Attributes
    ----------
    traj : mdtraj.Trajectory
        The structure, restricted to protein residues (waters, ions, ligands and
        terminal caps are removed).

    sequence : str
        One-letter sequence of the domain.

    sasa : numpy.ndarray
        Per-residue SASA in Angstrom^2.

    surface_vector : list of int
        1 for a surface residue and 0 for a buried one, per residue.

    surface_fraction : float
        Fraction of residues that are surface.

    surface_indices : list of int
        0-based indices of the surface residues.

    surface_positions : dict
        Maps each surface residue index to the centre of mass of its side chain (the
        alpha carbon for glycine), in nanometres.

    Raises
    ------
    ValueError
        If ``sasa_mode`` is not ``'v1'`` or ``'v2'``, if only one of ``start`` and
        ``end`` is given, if ``start``/``end`` are given without
        ``ignore_warnings=True``, if the file contains no protein residues, or if a
        residue has no standard amino acid equivalent and no entry in
        ``residue_overide_mapping``.
    """

    # ................................................................................
    #
    #
    def __init__(
        self,
        pdbfilename,
        start=None,
        end=None,
        probe_radius=1.4,
        residue_overide_mapping=None,
        surface_thresh=0.10,
        sasa_mode="v1",
        ignore_warnings=False,
        SASA_ONLY=False,
        SASA_override=None,
    ):
        """
        Load the structure, compute SASA and classify the surface.

        All parameters are documented in the class docstring.
        """
        if residue_overide_mapping is None:
            residue_overide_mapping = {}

        if sasa_mode not in ["v1", "v2"]:
            raise ValueError("sasa_mode must be v1 or v2")

        # safety feature for now...
        if start is not None and end is not None:
            message = (
                "Using non-standard start and end residues has not been thoroughly "
                "tested. We recommend excising the domain with "
                "extract_and_write_domains() and passing the new PDB file instead."
            )
            if ignore_warnings is False:
                raise ValueError(
                    message + " To use start/end anyway, set ignore_warnings=True."
                )
            warnings.warn(message)

        # parse pdbfile
        try:
            p = md.load_pdb(pdbfilename)
        except Exception as e:
            warnings.warn(
                f"mdtraj.load_pdb could not read {pdbfilename} ({e}); trying "
                "mdtraj.load instead."
            )
            p = md.load(pdbfilename)

        # handle start/end residues if passed
        if start is not None and end is not None:
            p = p.atom_slice(p.topology.select(f"resid {start} to {end}"))

        elif start is not None or end is not None:
            raise ValueError(
                "Either start and end passed or neither passed; one of start or end was passed"
            )

        # keep protein residues only, and drop terminal caps (which have no alpha
        # carbon). Waters, ions and ligands would otherwise be counted in the SASA
        # array but skipped when the sequence is built, silently misaligning every
        # downstream per-residue index.
        keep = [
            a.index
            for r in p.topology.residues
            if r.is_protein and any(a.name == "CA" for a in r.atoms)
            for a in r.atoms
        ]
        if len(keep) == 0:
            raise ValueError(f"No protein residues found in {pdbfilename}")
        if len(keep) < p.n_atoms:
            p = p.atom_slice(keep)

        # assign trajectory info
        self.traj = p

        # calculate SASA of residues;
        # 100* to convert from nm^2 to A^2, and *0.1 for probe radius to convert from A to nm
        self.sasa = (
            100
            * md.shrake_rupley(p, mode="residue", probe_radius=probe_radius * 0.1)[0]
        )

        # override SASA if passed
        if SASA_override is not None:
            for k in SASA_override:
                self.sasa[k] = SASA_override[k]

        # create full amino acid sequence. A residue we cannot name is an error
        # rather than a skipped position, because skipping it would misalign the
        # sequence against the per-residue SASA array.
        s = ""
        for r in p.topology.residues:
            aa = one_letter_code(r)
            if aa is None:
                aa = residue_overide_mapping.get(r.name)
            if aa is None:
                raise ValueError(
                    f"Residue {r} has no standard amino acid equivalent; pass "
                    "residue_overide_mapping={'"
                    + r.name
                    + "': '<one-letter code>'} to define it."
                )
            s = s + aa
        self.sequence = s

        # bail here
        if SASA_ONLY:
            return

        # find solvent surface res
        surface_vector = []
        surface_indices = []
        for i in range(len(self.sasa)):
            if sasa_mode == "v1":
                solvent_accessible = (
                    self.sasa[i] > surface_thresh * MAX_SASA_DATA[self.sequence[i]][0]
                )

            elif sasa_mode == "v2":
                solvent_accessible = self.sasa[i] > surface_thresh * (
                    MAX_SASA_DATA[self.sequence[i]][0]
                    + MAX_SASA_DATA[self.sequence[i]][1]
                )

            if solvent_accessible:
                surface_vector.append(1)
                surface_indices.append(i)
            else:
                surface_vector.append(0)

        self.surface_vector = surface_vector
        self.surface_fraction = sum(surface_vector) / len(surface_vector)
        self.surface_indices = surface_indices
        self.surface_positions = {}

        # if we're here cycle through surface residues
        for idx in self.surface_indices:
            # get atom indices for this residue

            atom_indices = p.topology.select(f"sidechain and resid {idx}")
            if len(atom_indices) == 0:
                atom_indices = p.topology.select(f"resid {idx} and name CA")

            self.surface_positions[idx] = md.compute_center_of_mass(
                p.atom_slice(atom_indices)
            )[0]

        # expensive things are computed on first access via the properties below
        self._all_shortest_paths = None
        self._surface_neighbours = None
        self._surface_neighbour_sequences = None
        self._surface_graph = None
        self._surface_distance_surface = None
        self._surface_distance_straight_line = None

    # ....................................................................
    #
    def __len__(self):
        """
        Number of residues in the domain.

        Returns
        -------
        int
            ``len(self.sequence)``.
        """
        return len(self.sequence)

    # ....................................................................
    #
    @property
    def all_shortest_paths(self):
        """
        Shortest paths through the surface neighbour graph.

        Returns
        -------
        dict
            Nested mapping ``{i: {j: [i, ..., j]}}`` of residue-index paths between
            every pair of connected surface residues (networkx Dijkstra paths).
        """
        if self._all_shortest_paths is None:
            self._all_shortest_paths = dict(
                nx.all_pairs_dijkstra_path(self.surface_graph, weight="weight")
            )
        return self._all_shortest_paths

    # ....................................................................
    #
    @property
    def surface_graph(self):
        """
        Graph connecting surface residues that are within the neighbour distance.

        Returns
        -------
        networkx.Graph
            Nodes are surface residue indices; edges join residues whose side-chain
            centres are within the neighbour distance (9 Angstrom unless
            get_nearest_neighbour_res() was called with another value) and carry that
            distance, in Angstrom, as their ``weight``.
        """
        if self._surface_graph is None:
            self.get_nearest_neighbour_res()
        return self._surface_graph

    # ....................................................................
    #
    @property
    def surface_neighbours(self):
        """
        Neighbours of each surface residue.

        Returns
        -------
        dict
            Maps each surface residue index to a list of ``[index, distance]`` pairs
            (distance in Angstrom) for every surface residue within the neighbour
            distance, including the residue itself at distance zero.
        """
        if self._surface_neighbours is None:
            self.get_nearest_neighbour_res()
        return self._surface_neighbours

    # ....................................................................
    #
    @property
    def surface_neighbour_sequences(self):
        """
        One-letter sequence of each surface residue's neighbours.

        Returns
        -------
        dict
            Maps each surface residue index to the concatenated one-letter codes of
            its neighbours (itself included), in the order of surface_neighbours.
        """
        if self._surface_neighbour_sequences is None:
            self.get_nearest_neighbour_res()
        return self._surface_neighbour_sequences

    # ....................................................................
    #
    @property
    def surface_distance_surface(self):
        """
        Over-the-surface distances between surface residues.

        Returns
        -------
        dict
            Nested mapping ``{i: {j: distance}}`` of shortest-path lengths through
            the surface graph, in Angstrom. Residues in different connected
            components of the graph have no entry for each other.
        """
        if self._surface_distance_surface is None:
            self.get_nearest_neighbour_res()
        return self._surface_distance_surface

    # ....................................................................
    #
    @property
    def surface_distance_straight_line(self):
        """
        Straight-line distances between surface residues.

        Returns
        -------
        dict
            Nested mapping ``{i: {j: distance}}`` of Euclidean distances between
            side-chain centres, in Angstrom, for every pair of surface residues.
        """
        if self._surface_distance_straight_line is None:
            self.get_nearest_neighbour_res()
        return self._surface_distance_straight_line

    # ....................................................................
    #
    def __str__(self):
        """
        Return a short description of the domain.

        Returns
        -------
        str
            ``"FoldedDomain object with <n> residues"``.
        """
        return f"FoldedDomain object with {len(self.sequence)} residues"

    # ....................................................................
    #
    def __repr__(self):
        """
        Return the same short description as str().

        Returns
        -------
        str
            ``"FoldedDomain object with <n> residues"``.
        """
        return str(self)

    # ................................................................................
    #
    #
    def _neighbour_data(self, distance_thresh):
        """
        Build the surface neighbour data for a given neighbour distance.

        This computes everything get_nearest_neighbour_res() stores, but returns it
        instead of caching it on the object, so a method that needs a different
        neighbour distance can use it without changing what the properties report.

        Parameters
        ----------
        distance_thresh : float
            Neighbour distance in Angstrom: two surface residues are neighbours if
            their side-chain centres are closer than this.

        Returns
        -------
        dict
            ``neighbours`` (index -> list of ``[index, distance]``, the residue
            itself included), ``graph`` (networkx.Graph), ``surface_distance``
            (over-the-surface shortest-path lengths), ``straight_line_distance``
            (Euclidean distances) and ``neighbour_sequences`` (index -> neighbour
            one-letter string). Distances are in Angstrom.
        """
        # positions are stored in nm; distances below are reported in Angstrom
        positions = np.array([self.surface_positions[i] for i in self.surface_indices])
        idx = list(self.surface_indices)
        dist_matrix = distance.cdist(positions, positions, "euclidean") * 10.0

        straight_line = {
            r1: {r2: dist_matrix[a][b] for b, r2 in enumerate(idx)}
            for a, r1 in enumerate(idx)
        }

        neighbours = {}
        for a, r1 in enumerate(idx):
            close = np.where(dist_matrix[a] < distance_thresh)[0]
            neighbours[r1] = [[idx[b], dist_matrix[a][b]] for b in close]

        graph = nx.Graph()
        for r1, connections in neighbours.items():
            for r2, d in connections:
                graph.add_edge(r1, r2, weight=d)

        return {
            "neighbours": neighbours,
            "graph": graph,
            "surface_distance": dict(
                nx.all_pairs_dijkstra_path_length(graph, weight="weight")
            ),
            "straight_line_distance": straight_line,
            "neighbour_sequences": {
                r1: "".join(self.sequence[r2] for r2, _ in conn)
                for r1, conn in neighbours.items()
            },
        }

    # ................................................................................
    #
    #
    def get_nearest_neighbour_res(self, distance_thresh=9.0):
        """
        Find the neighbours of every surface residue and rebuild the surface graph.

        Two surface residues are neighbours if their side-chain centres are within
        ``distance_thresh`` of each other. This sets what the surface_neighbours,
        surface_neighbour_sequences, surface_graph, surface_distance_surface,
        surface_distance_straight_line and all_shortest_paths properties report; the
        properties call it with the default distance on first access, and calling it
        again with a different distance replaces all of them.

        Parameters
        ----------
        distance_thresh : float, optional
            Neighbour distance in Angstrom. Default 9.0.

        Returns
        -------
        None
        """
        data = self._neighbour_data(distance_thresh)
        self._surface_graph = data["graph"]
        self._surface_neighbours = data["neighbours"]
        self._surface_distance_surface = data["surface_distance"]
        self._surface_distance_straight_line = data["straight_line_distance"]
        self._surface_neighbour_sequences = data["neighbour_sequences"]

        # the shortest paths belong to the old graph
        self._all_shortest_paths = None

    # ................................................................................
    #
    #
    def __extract_residues(self, inseq, targets):
        """
        Internal function that removes all residues in a list that
        are in a target list, returning the modified list and the
        removed elements.

        Parameters
        ----------
        inseq : str or list of str
            Input residues.

        targets : collection of str
            Residues to remove.

        Returns
        -------
        tuple of (list, list)
            The residues that were kept, in their original order, and the residues
            that were removed, in the reverse of the order they appeared.

        """

        inseq = list(inseq)

        removed_elements = []
        for i in range(len(inseq) - 1, -1, -1):
            if inseq[i] in targets:
                removed_elements.append(inseq.pop(i))

        return inseq, removed_elements

    # ................................................................................
    #
    #
    def calculate_surface_epsilon(self, input_sequence, IMCObject):
        """
        Calculate a surface epsilon for every surface residue against an input sequence.

        For each surface residue:

        1. Take the residue and its neighbours (the surface residues within the
           neighbour distance, see get_nearest_neighbour_res(); the residue itself is
           included).
        2. Reorder that neighbour string so residues of the centre residue's chemical
           class sit next to each other at its start: aliphatics (A, V, I, L, M) if the
           centre is aliphatic, acidic residues if it is acidic, basic residues if it is
           basic, and no reordering otherwise. This lets the sequence-based charge and
           aliphatic weighting treat them as a local cluster.
        3. Compute the FINCHES epsilon between the neighbour string and the whole input
           sequence and divide by the length of the neighbour string. The result is the
           mean, over the neighbour residues, of each residue's mean interaction with the
           input sequence.

        Because the whole input sequence enters every value, this reports how each patch
        of surface interacts with the input sequence on average; for a surface:IDR
        intermap that resolves regions of the IDR, use
        calculate_idr_surface_patch_interactions() or StructureSurface.

        Parameters
        ----------
        input_sequence : str
            Amino acid sequence to score the surface against.

        IMCObject : finches.epsilon_calculation.InteractionMatrixConstructor
            The interaction-matrix constructor that provides the energetics, e.g. the
            IMC_object attribute of an Mpipi_frontend or CALVADOS_frontend.

        Returns
        -------
        dict
            Maps each surface residue index (0-based) to a list of three elements: the
            residue's one-letter code, the reordered neighbour string, and the surface
            epsilon (negative is attractive).
        """

        surface_eps = {}

        for idx in self.surface_indices:
            center_resid = self.sequence[idx]

            # the residues neighbouring this one. This includes the residue itself,
            # since it is within any distance threshold of itself, so the local
            # environment string carries the centre residue's own chemistry too.
            local_seq = [self.sequence[n] for n, _ in self.surface_neighbours[idx]]

            # group residues of the centre's chemical class next to each other, so
            # the sequence-based charge and aliphatic weighting sees them as a cluster
            for group in (ALIPHATIC_RESIDUES, NEGATIVE_RESIDUES, POSITIVE_RESIDUES):
                if center_resid in group:
                    local_seq, removed = self.__extract_residues(local_seq, group)
                    reordered_seq = "".join(removed) + "".join(local_seq)
                    break
            else:
                reordered_seq = "".join(local_seq)

            tmp = IMCObject.calculate_epsilon_value(
                reordered_seq, input_sequence
            ) / len(reordered_seq)

            surface_eps[idx] = [center_resid, reordered_seq, tmp]

        return surface_eps

    # ................................................................................
    #
    #
    def calculate_mean_surface_epsilon(self, input_sequence, IMCObject):
        """
        Mean surface epsilon over all surface residues.

        Parameters
        ----------
        input_sequence : str
            Amino acid sequence to score the surface against.

        IMCObject : finches.epsilon_calculation.InteractionMatrixConstructor
            The interaction-matrix constructor that provides the energetics, e.g. the
            IMC_object attribute of an Mpipi_frontend or CALVADOS_frontend.

        Returns
        -------
        float
            The mean of the values returned by calculate_surface_epsilon(), a measure of
            how the domain's surface as a whole interacts with the input sequence.
        """

        surface_eps = self.calculate_surface_epsilon(input_sequence, IMCObject)

        return np.mean([i[2] for i in surface_eps.values()])

    # ................................................................................
    #
    #
    def calculate_attractive_surface_epsilon(
        self, input_sequence, IMCObject, threshold=0
    ):
        """
        Surface epsilon values of the surface residues that are attractive.

        Computes calculate_surface_epsilon() and keeps the values below ``threshold``. To
        relate the values to residue positions use calculate_surface_epsilon() directly.

        Parameters
        ----------
        input_sequence : str
            Amino acid sequence to score the surface against.

        IMCObject : finches.epsilon_calculation.InteractionMatrixConstructor
            The interaction-matrix constructor that provides the energetics, e.g. the
            IMC_object attribute of an Mpipi_frontend or CALVADOS_frontend.

        threshold : float, optional
            Values strictly below this are kept. Default 0.

        Returns
        -------
        list of float
            The attractive surface epsilon values, in surface-residue order.
        """

        surface_eps = self.calculate_surface_epsilon(input_sequence, IMCObject)

        return [i[2] for i in surface_eps.values() if i[2] < threshold]

    # ................................................................................
    #
    #
    def calculate_repulsive_surface_epsilon(
        self, input_sequence, IMCObject, threshold=0
    ):
        """
        Surface epsilon values of the surface residues that are not attractive.

        Computes calculate_surface_epsilon() and keeps the values at or above
        ``threshold``, the complement of calculate_attractive_surface_epsilon() for the
        same threshold. To relate the values to residue positions use
        calculate_surface_epsilon() directly.

        Parameters
        ----------
        input_sequence : str
            Amino acid sequence to score the surface against.

        IMCObject : finches.epsilon_calculation.InteractionMatrixConstructor
            The interaction-matrix constructor that provides the energetics, e.g. the
            IMC_object attribute of an Mpipi_frontend or CALVADOS_frontend.

        threshold : float, optional
            Values at or above this are kept. Default 0.

        Returns
        -------
        list of float
            The repulsive surface epsilon values, in surface-residue order.
        """

        surface_eps = self.calculate_surface_epsilon(input_sequence, IMCObject)

        return [i[2] for i in surface_eps.values() if i[2] >= threshold]

    # ................................................................................
    #
    #
    def calculate_IWD(
        self,
        residues,
        positions=None,
        distance_mode="surface",
        calculate_null=False,
        number_null_iterations=100,
    ):
        """
        Inverse weighted distance (IWD) between a set of surface residues.

        The IWD is the mean over all pairs of the selected surface residues of
        1 / distance (equation 1 of Schueler-Furman & Baker, 2003), so it is larger when
        the residues are clustered. Distances are either over the surface (shortest paths
        through the surface graph) or straight-line. Pairs in different connected
        components of the surface graph have no over-the-surface path and contribute
        zero. To judge whether a value is unusual, a null distribution can be built from
        the same number of randomly chosen surface residues. This has not been
        extensively tested.

        Parameters
        ----------
        residues : collection of str
            One-letter codes of the residue types to include (e.g. ``['K', 'R']``).
            Ignored if ``positions`` is given.

        positions : list of int, optional
            0-based residue indices to include instead of selecting by type; indices
            that are not surface residues are skipped. Default None.

        distance_mode : str, optional
            ``'surface'`` (default) or ``'straight line'``.

        calculate_null : bool, optional
            If True, also compute a null distribution. Default False.

        number_null_iterations : int, optional
            Number of random draws for the null distribution. Default 100.

        Returns
        -------
        list
            ``[iwd, n_residues, null]``: the IWD (1/Angstrom), the number of surface
            residues used, and either the list of null IWD values (if
            ``calculate_null``) or -1. If fewer than two surface residues are selected,
            ``[None, None, None]``.

        Raises
        ------
        ValueError
            If ``distance_mode`` is not recognised.
        """

        # define distance function based on distance mode. Two residues in different
        # connected components of the surface graph have no over-the-surface path, so
        # their distance is infinite and they contribute nothing to the IWD
        if distance_mode == "surface":

            def dist(i1, i2):
                return self.surface_distance_surface[i1].get(i2, np.inf)
        elif distance_mode == "straight line":

            def dist(i1, i2):
                return self.surface_distance_straight_line[i1][i2]
        else:
            raise ValueError(
                'distance mode must be either "surface" or "straight line"'
            )

        relevant_indices = []

        # if we have a list of positions, use those instead of the residue selector list
        if positions is not None:
            for idx in positions:
                if idx in self.surface_indices:
                    relevant_indices.append(idx)

        # otherwise, use the residue selector list
        else:
            for idx, res in enumerate(self.sequence):
                if res in residues:
                    if idx in self.surface_indices:
                        relevant_indices.append(idx)

        if len(relevant_indices) < 2:
            return [None, None, None]

        # calculate the inverse weighted distance as defined by
        # equation 1 in Schueler-Furman & Baker, 2003
        summation = 0
        npairs = 0
        for i1 in range(0, len(relevant_indices) - 1):
            # index 1
            idx1 = relevant_indices[i1]

            for i2 in range(i1 + 1, len(relevant_indices)):
                # index 2
                idx2 = relevant_indices[i2]

                summation = summation + 1 / dist(idx1, idx2)
                npairs = npairs + 1

        iwd = summation / npairs

        if calculate_null is False:
            return [iwd, len(relevant_indices), -1]

        else:
            iwd_null = []
            for i in range(number_null_iterations):
                # randomly sample positions from the surface
                random_positions = np.random.choice(
                    self.surface_indices, len(relevant_indices), replace=False
                )
                iwd_null.append(
                    self.calculate_IWD(
                        [], positions=random_positions, distance_mode=distance_mode
                    )[0]
                )

            return [iwd, len(relevant_indices), iwd_null]

    # ................................................................................
    #
    #
    def write_SASA_vis_file(self, filename="sasa_binary.txt"):
        """
        Write a text file flagging surface (1) and buried (0) residues.

        Each line is ``<residue number> A <0 or 1>`` with 1-based residue numbers and a
        fixed chain label A, a format that structure-colouring scripts can read.

        Parameters
        ----------
        filename : str, optional
            Output path. Default ``'sasa_binary.txt'``.

        Returns
        -------
        None
        """

        with open(filename, "w") as fh:
            for i in range(len(self.sasa)):
                fh.write(f"{i + 1} A {self.surface_vector[i]}\n")

    # ................................................................................
    #
    #
    def write_epsilon_vis_file(self, surface_epsilon, filename="surface_epsilon.txt"):
        """
        Write a text file of per-residue surface epsilon values.

        Each line is ``<residue number> A <epsilon>``, with 1-based residue numbers, a
        fixed chain label A, and epsilon rounded to three decimals, a format that
        structure-colouring scripts can read. Lines run from the first to the last
        surface residue in surface_epsilon; residues in between that are not in it
        (buried residues) are written with a value of 0.

        Parameters
        ----------
        surface_epsilon : dict
            The dictionary returned by calculate_surface_epsilon().

        filename : str, optional
            Output path. Default ``'surface_epsilon.txt'``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``surface_epsilon`` is None.
        """

        if surface_epsilon is None:
            raise ValueError(
                "Surface epsilon has not been calculated. Run calculate_surface_epsilon first."
            )

        max_idx = max(surface_epsilon)
        min_idx = min(surface_epsilon)

        with open(filename, "w") as fh:
            for i in range(min_idx, max_idx + 1):
                if i not in surface_epsilon:
                    fh.write(f"{i + 1} A 0\n")
                else:
                    fh.write(f"{i + 1} A {round(surface_epsilon[i][2], 3)}\n")

    def calculate_idr_surface_patch_interactions(
        self, interacting_sequence, IMCObject, idr_tile_size=31, patch_radius=12
    ):
        """
        Calculate the interaction between an IDR and the surface patches of the
        folded domain, giving a surface:IDR intermap.

        A patch is a surface residue together with every surface residue within
        patch_radius of it (straight-line distance between side-chain centres of
        mass, including the centre residue itself). Each patch is scored against
        every sliding window of the IDR with the standard FINCHES epsilon, so the
        rows of the returned matrix are on the same scale as an IDR:IDR intermap.

        The patches are built with patch_radius directly and do not change the
        neighbour graph that the surface_neighbours and related properties report.

        For a structure-aware version that decomposes the structure into IDRs and
        folded domains and uses a contiguous-surface footprint, see
        finches.utils.structure_surface.StructureSurface.surface_vs_idr_matrix().

        Parameters
        -----------
        interacting_sequence : str
            the sequence that interacts with the protein in the PDB as a string

        IMCObject: IMCObject
            IMCObject that contains the epsilon calculate. The IMCObject is an initialized
            FINCHES-derived object that enables epsilon calculations. It can be found in
            the Mpipi_frontend or CALAVDOS_frontend objects, as an IMC_object variable.

        idr_tile_size : int
            Size of the sliding window over the IDR. Must be odd and no longer than
            the IDR. Default is 31, for consistency with IDR:IDR intermaps.

        patch_radius : float
            the radius from the center of the patch to allow residues to contribute
            to the patch. Default is 12 Angstroms.

        Returns
        --------
        list
            [0] dict keyed by surface residue index (0-based). Each value is a dict
                with 'mean_epsilon' (the FINCHES epsilon between the whole patch and
                the whole IDR, i.e. summed over the patch residues and averaged over
                the IDR), 'interacting_residues' (the patch sequence),
                'residue_indices', 'resinds_with_dist' ([index, distance] pairs) and
                'idr_epsilon_vector' (the patch's epsilon against each IDR window).

            [1] numpy array of shape (n_residues, n_windows): the surface:IDR
                intermap, one row per residue in the folded domain (rows for buried
                residues are np.nan) and one column per IDR window.

            [2] numpy array of length n_windows: the mean over all surface patches
                of the epsilon against each IDR window.

        """

        if (idr_tile_size % 2) == 0:
            raise ValueError("IDR tile window size must be an odd integer.")

        if idr_tile_size > len(interacting_sequence):
            raise ValueError(
                "IDR tile window size must be no longer than the provided sequence."
            )

        half_window = int((idr_tile_size - 1) / 2)

        # patches at patch_radius, without touching the cached neighbour graph
        neighbors = self._neighbour_data(patch_radius)["neighbours"]

        # get sequence
        seq = self.sequence

        # dict to hold patches
        interaction_dict = {}

        # now iterate over all possible patches
        for patch_ind_center in neighbors:
            resinds = neighbors[patch_ind_center]

            patch_mean_eps = sum(
                [
                    IMCObject.calculate_epsilon_value(interacting_sequence, seq[a])
                    for a, dist in resinds
                ]
            ) / len(interacting_sequence)

            interacting_residues = "".join([seq[a] for a, dist in resinds])
            residue_indices = [a for a, n in resinds]
            interaction_dict[patch_ind_center] = {
                "mean_epsilon": patch_mean_eps,
                "interacting_residues": interacting_residues,
                "residue_indices": residue_indices,
                "resinds_with_dist": resinds,
            }

        for i in interaction_dict.keys():
            res_data = interaction_dict[i]

            tmp = []
            for j in range(half_window, len(interacting_sequence) - half_window):
                tmp.append(
                    IMCObject.calculate_epsilon_value(
                        interacting_sequence[(j - half_window) : (j + half_window + 1)],
                        res_data["interacting_residues"],
                    )
                )

            interaction_dict[i]["idr_epsilon_vector"] = tmp

        # pre-initailize the empty vector
        empty_vector = np.array(
            [np.nan] * len(list(interaction_dict.values())[0]["idr_epsilon_vector"])
        )

        collect_epsilon_vectors = []

        for i in range(len(self)):
            # if residue i is a surface residue
            if interaction_dict.get(i):
                collect_epsilon_vectors.append(
                    np.array(interaction_dict[i]["idr_epsilon_vector"])
                )
            else:
                collect_epsilon_vectors.append(empty_vector)

        vector_mean = np.array(
            [x["idr_epsilon_vector"] for x in interaction_dict.values()]
        ).mean(axis=0)

        return [interaction_dict, np.array(collect_epsilon_vectors), vector_mean]


# < end of class> - note we define this here because we actually want to alias the old
# name to the new name....
FoldeDomain = FoldedDomain


# ................................................................................
#
#
def extract_and_write_domains(
    pdb_file: str, outfile: str, start: int, end: int, reset_indices: bool = True
) -> None:
    """
    This function will extract a domain from a pdb file and write it
    to a new file. Note this function will renumber the residues and atoms
    to start at 1 assuming reset_indices is True.

    Parameters
    ----------
    pdb_file : str
        The path to the pdb file.

    outfile : str
        The path to the output file.

    start : int
        The start residue of the domain, as a 0-based mdtraj residue index
        (inclusive).

    end : int
        The end residue of the domain, as a 0-based mdtraj residue index
        (inclusive).

    reset_indices : bool
        If True, the function will reset the
        residue and atom indices to start at 1.

    Returns
    -------
    None

    """

    # read pdb file
    p = md.load(pdb_file)

    # extract domain
    p = p.atom_slice(p.topology.select(f"resid {start} to {end}"))

    # if we want to reset the indices to star at 1...
    if reset_indices:
        # recode residue indices
        for i, residue in enumerate(p.topology.residues):
            residue.index = i + 1  # Setting new residue index starting from 1
            residue.resSeq = i + 1  # Setting new residue index starting from 1

        # recode atom indices
        for i, atom in enumerate(p.topology.atoms):
            atom.index = i + 1
            atom.serial = i + 1

    # write pdb file
    p.save_pdb(outfile)


# def inter_domain_epsilon(fd1, fd2):
#    for
