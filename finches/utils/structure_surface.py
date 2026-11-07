"""
Structure-aware surface interaction analysis for FINCHES.

.. warning::

   **This module is in active development.** The API, the default parameters, and the
   numbers it produces are all still moving, and none of it should be treated as stable
   or as validated science yet. In particular the polymer-reach parameters
   (``reach_b``, ``reach_nu``), ``contact_radius``, and the surface-classification and
   occlusion thresholds are *uncalibrated* - they are physically-motivated defaults
   rather than fitted values, and ``reach_b`` sets the contact-footprint radius directly
   and has not yet had a sensitivity sweep. Expect the defaults, and therefore the
   output, to change.

This module provides the :class:`StructureSurface` class, which reads a single- or
multi-chain protein structure (PDB or mmCIF/PDBx) and supports a structure-aware
interaction workflow:

1. Decompose each chain into intrinsically disordered regions (IDRs) and folded
   domains (FDs), including IDR residues that are missing from the coordinates.
   Stretches that metapredict predicts to be disordered but that pack against a
   folded domain in the structure (for example a helix that folds on binding a
   partner chain) are treated as folded.
2. Within folded domains only, classify residues as solvent-exposed (surface) or
   buried (IDRs are excluded from the SASA calculation so dangling tails do not
   occlude the folded-domain surface).
3. Build a *contiguous-surface* neighbourhood graph (a "net") over the surface
   residues, so two residues are only neighbours if they share a real, continuous
   surface (not merely close in space across the interior of the domain).
4. Use FINCHES to compute context-aware interaction scores for each surface residue
   against an IDR sequence (of arbitrary length) or another surface residue,
   replicating the FINCHES local charge/aliphatic weighting but over the surface
   neighbourhood rather than the linear sequence.
5. Use the folded-domain geometry to limit where an IDR can physically reach, using a
   simple polymer model ``R(i) = b * i**nu`` (defaults ``nu=0.54``, ``b=5`` Angstrom).
6. Build and plot a surface:IDR intermap (:meth:`StructureSurface.surface_vs_idr_matrix`
   and :meth:`StructureSurface.plot_surface_vs_idr_matrix`) that sits on the same scale
   as a FINCHES IDR:IDR intermap, with buried residues greyed out.

This module supersedes the patch mode in ``finches.utils.folded_domain_utils.FoldedDomain``,
which is kept so that earlier results can be reproduced. The residue-class definitions
and the charge and aliphatic weighting rules come from
``finches.parsing_aminoacid_sequences``, so the structure-aware scores use exactly the
same chemistry as an IDR:IDR calculation.

By: Alex Holehouse & contributors
"""

import os
import warnings

import numpy as np
import mdtraj as md
import networkx as nx
from scipy.spatial import cKDTree
import metapredict as meta

from finches.parsing_aminoacid_sequences import (
    ALIPHATIC_RESIDUES,
    CHARGED_RESIDUES,
    NEGATIVE_RESIDUES,
    POSITIVE_RESIDUES,
    aliphatic_weight_from_levels,
    charge_weight_from_counts,
    get_aliphatic_groups,
    get_charge_window_counts,
)
from finches.utils.folded_domain_utils import MAX_SASA_DATA, one_letter_code

# file extensions that should be read with mdtraj's dedicated PDB reader; everything
# else (notably .cif/.mmcif/.pdbx) is dispatched through md.load, which selects the
# correct reader from the extension.
_PDB_EXTENSIONS = (".pdb", ".pdb.gz", ".ent", ".ent.gz")

# default polymer-reach parameters (simple self-avoiding-walk-like scaling)
DEFAULT_REACH_B = 5.0  # Angstrom
DEFAULT_REACH_NU = 0.54

# Radius of a single residue's contact shell, in Angstrom. An IDR residue sitting over
# the surface is not a point: it touches everything within roughly one side-chain
# contact distance, so this floors the reach radius (see surface_vs_idr_matrix). The
# default matches the median nearest-neighbour over-the-surface distance between
# surface residues, i.e. one shell of touching neighbours.
DEFAULT_CONTACT_RADIUS = 6.0  # Angstrom

# Stretches that metapredict calls disordered but that are packed in the structure.
# metapredict works from sequence alone, so a segment that only folds in context (a
# helix packed against a partner chain, or the edge of a domain that a predicted
# boundary cuts off) is called disordered even though the structure shows it packed.
# Such a stretch is treated as folded when its residues touch residues of the
# predicted folded domains (see StructureSurface._rescue_packed_idr_residues). Like the
# other thresholds in this module these are uncalibrated: they were chosen by
# inspecting AlphaFold models, where the stretches they rescue have high pLDDT and
# floppy tails are left alone.
PACKING_CONTACT_CUTOFF = 4.5  # Angstrom, heavy atom to heavy atom
# partners closer than this along the same chain are local (helical i, i+4 contacts and
# the residues bonded across a domain boundary) and do not count as packing
PACKING_MIN_SEQUENCE_SEPARATION = 5  # residues
# partner counts are averaged over a window of this many residues (odd); gaps shorter
# than the window between packed residues are closed
PACKING_WINDOW = 9  # residues
# a residue is packed when its window averages at least this many distinct
# folded-domain partners per residue
PACKING_MIN_PARTNERS = 1.0
# an isolated packed stretch shorter than this is ignored as noise
PACKING_MIN_RUN = 5  # residues


def _load_structure(path):
    """
    Load a protein structure from a PDB or mmCIF/PDBx file with mdtraj.

    PDB files are read with ``mdtraj.load_pdb`` and all other extensions (including
    ``.cif``/``.mmcif``/``.pdbx``) are dispatched through ``mdtraj.load``, which
    selects the appropriate reader. If ``mdtraj.load_pdb`` fails on a file with a PDB
    extension, ``mdtraj.load`` is tried as a fallback.

    Parameters
    ----------
    path : str
        Path to the structure file.

    Returns
    -------
    mdtraj.Trajectory
        The loaded structure (single frame for a static model).
    """
    if str(path).lower().endswith(_PDB_EXTENSIONS):
        try:
            return md.load_pdb(path)
        except Exception:
            # fall back to letting mdtraj pick the reader itself
            pass
    return md.load(path)


def _windowed_mean(values, window):
    """
    Mean of each value and its neighbours, truncated at the ends of the array.

    Parameters
    ----------
    values : numpy.ndarray
        1D array of per-residue values.
    window : int
        Odd window length in residues.

    Returns
    -------
    numpy.ndarray
        Array the same length as ``values``; element ``i`` is the mean of
        ``values[i - window // 2 : i + window // 2 + 1]``, clipped to the array, so
        residues near the ends average over fewer neighbours rather than over padding.
    """
    half_window = window // 2
    means = np.empty(len(values))
    for i in range(len(values)):
        lo = max(0, i - half_window)
        hi = min(len(values), i + half_window + 1)
        means[i] = values[lo:hi].mean()
    return means


def _fill_short_gaps(packed, modeled, n_flank_folded, c_flank_folded, max_gap):
    """
    Close short gaps in a packed mask along one IDR run.

    A gap of unpacked residues shorter than ``max_gap`` is filled when it lies between
    two packed residues, or between a packed residue and the folded domain flanking the
    run, and every residue in it is modeled (missing residues are never folded).

    Parameters
    ----------
    packed : numpy.ndarray of bool
        Packed flag for each residue of the run, in sequence order.
    modeled : numpy.ndarray of bool
        Whether each residue of the run has coordinates.
    n_flank_folded, c_flank_folded : bool
        Whether the run is flanked by a modeled folded residue at its N or C end.
    max_gap : int
        Gaps strictly shorter than this are filled.

    Returns
    -------
    numpy.ndarray of bool
        A new mask with the short gaps filled. A run with no packed residue is
        returned unchanged, so two folded flanks alone never fill a run.
    """
    filled = packed.copy()
    anchors = list(np.flatnonzero(packed))
    if not anchors:
        return filled
    # the flanking folded residues sit just outside the run, at -1 and len(run)
    if n_flank_folded:
        anchors = [-1] + anchors
    if c_flank_folded:
        anchors = anchors + [len(packed)]
    for left, right in zip(anchors[:-1], anchors[1:]):
        gap = right - left - 1
        if 0 < gap < max_gap and modeled[left + 1 : right].all():
            filled[left + 1 : right] = True
    return filled


def _drop_short_isolated_runs(packed, n_flank_folded, c_flank_folded, min_run):
    """
    Remove packed stretches that are too short and do not touch a folded domain.

    Parameters
    ----------
    packed : numpy.ndarray of bool
        Packed flag for each residue of the run, in sequence order.
    n_flank_folded, c_flank_folded : bool
        Whether the run is flanked by a modeled folded residue at its N or C end. A
        stretch reaching that end extends the folded domain, so it is kept whatever
        its length.
    min_run : int
        Isolated stretches shorter than this are dropped.

    Returns
    -------
    numpy.ndarray of bool
        A new mask without the short isolated stretches.
    """
    kept = packed.copy()
    n_residues = len(packed)
    i = 0
    while i < n_residues:
        if not packed[i]:
            i += 1
            continue
        j = i
        while j < n_residues and packed[j]:
            j += 1
        touches_folded = (i == 0 and n_flank_folded) or (
            j == n_residues and c_flank_folded
        )
        if j - i < min_run and not touches_folded:
            kept[i:j] = False
        i = j
    return kept


class Residue:
    """
    Lightweight per-residue record used internally by :class:`StructureSurface`.

    Attributes
    ----------
    chain_index : int
        0-based mdtraj chain index.
    res_seq : int
        Author residue number (PDB ``resSeq``). A residue that is missing from the
        coordinates carries the number it would have had: its 1-based position in a
        provided sequence, or the ``resSeq`` that fills the numbering gap when the
        sequence is reconstructed from the structure.
    seq_index : int
        0-based position of this residue in the full chain sequence.
    one_letter : str
        One-letter amino acid code.
    modeled : bool
        True if the residue has coordinates in the structure.
    domain : str
        Either ``'folded'`` or ``'idr'``.
    packed_rescue : bool
        True if metapredict predicts the residue to be disordered but it is treated as
        folded because it packs against a folded domain in the structure (see
        :meth:`StructureSurface._rescue_packed_idr_residues`).
    md_index : int or None
        Global mdtraj residue index (None for missing residues).
    position : numpy.ndarray or None
        Side-chain centre-of-mass coordinate in Angstrom (None unless the residue is a
        modeled, solvent-exposed folded-domain residue).
    surface : bool
        True if a folded, modeled residue is solvent-exposed. Always False for
        IDR / buried / missing residues.
    """

    __slots__ = (
        "chain_index",
        "res_seq",
        "seq_index",
        "one_letter",
        "modeled",
        "domain",
        "packed_rescue",
        "md_index",
        "position",
        "surface",
    )

    def __init__(self, chain_index, res_seq, seq_index, one_letter, modeled):
        """
        Create a residue record.

        Newly created records default to ``domain='idr'``, ``packed_rescue=False``,
        ``surface=False``, ``md_index=None`` and ``position=None``; these are populated by
        :class:`StructureSurface` during decomposition and surface classification.

        Parameters
        ----------
        chain_index : int
            0-based mdtraj chain index.
        res_seq : int
            Author residue number (PDB ``resSeq``).
        seq_index : int
            0-based position of the residue in the full chain sequence.
        one_letter : str
            One-letter amino acid code.
        modeled : bool
            Whether the residue has coordinates in the structure.
        """
        self.chain_index = chain_index
        self.res_seq = res_seq
        self.seq_index = seq_index
        self.one_letter = one_letter
        self.modeled = modeled
        self.domain = "idr"
        self.packed_rescue = False
        self.md_index = None
        self.position = None
        self.surface = False

    @property
    def key(self):
        """
        Stable identifier for the residue.

        Returns
        -------
        tuple of (int, int)
            ``(chain_index, res_seq)``, used as the key throughout
            :class:`StructureSurface`.
        """
        return (self.chain_index, self.res_seq)

    def __repr__(self):
        """
        Return a concise developer-readable representation of the residue.

        Returns
        -------
        str
            A string of the form
            ``Residue(chain=<i>, resSeq=<n>, <AA>, <domain>[/surface])``.
        """
        return (
            f"Residue(chain={self.chain_index}, resSeq={self.res_seq}, "
            f"{self.one_letter}, {self.domain}{'/surface' if self.surface else ''})"
        )


class StructureSurface:
    """
    Structure-aware surface interaction analysis for a folded protein assembly.

    Reads a single- or multi-chain structure (PDB or mmCIF/PDBx), decomposes each
    chain into folded domains and IDRs, identifies the contiguous folded-domain
    surface, and provides FINCHES-based interaction scoring (optionally constrained by
    how far an IDR can physically reach). See the module docstring for the full
    workflow.

    Parameters
    ----------
    pdbfilename : str
        Path to the structure file. Both PDB (``.pdb``) and mmCIF/PDBx
        (``.cif``/``.mmcif``/``.pdbx``) are supported.

    frontend : finches frontend object
        A FINCHES frontend (e.g. ``Mpipi_frontend()`` or ``CALVADOS_frontend()``)
        providing the interaction energetics. Its ``IMC_object`` is used internally
        for all interaction scoring, so callers never have to handle the
        interaction-matrix constructor directly. (An ``InteractionMatrixConstructor``
        may also be passed directly.)

    sequences : dict, list, str or None, optional
        Full per-chain amino acid sequences, used to recover IDR residues that are
        missing from the coordinates. May be:

        - a ``dict`` mapping 0-based chain index -> sequence string,
        - a ``list`` of sequences (one per protein chain, in chain order),
        - a single ``str`` (only valid for a single-chain structure),
        - ``None`` (default) to reconstruct each chain sequence from the modeled
          residues, filling any ``resSeq`` numbering gaps as (missing) IDR residues.

        When provided, a chain's modeled residues are aligned to the sequence by
        ``resSeq`` (assumed 1-based into the sequence) and residue identities are
        validated.

    probe_radius : float, optional
        SASA probe radius in Angstrom. Default 1.4.

    surface_thresh : float, optional
        Fraction of maximum SASA above which a residue is called solvent-exposed.
        Default 0.10.

    sasa_mode : str, optional
        ``'v1'`` (compare to max side-chain SASA) or ``'v2'`` (compare to max
        side-chain + backbone SASA). Default ``'v1'`` (see
        :class:`finches.utils.folded_domain_utils.FoldedDomain`).

    net_distance_thresh : float, optional
        Maximum Angstrom distance between two surface residues' side-chain
        centres-of-mass for them to be candidate neighbours. Default 9.0.

    occlusion_radius : float, optional
        Radius in Angstrom used to gather neighbouring heavy atoms around a sampled
        point when testing whether a candidate edge passes through the buried core.
        Default 5.0.

    occlusion_surround_thresh : float, optional
        A sampled point is considered buried when the mean of the unit vectors to
        the heavy atoms within ``occlusion_radius`` has magnitude below this value
        (i.e. atoms surround the point on all sides). Larger values prune more
        aggressively. Default 0.3.

    occlusion_sample_spacing : float, optional
        Spacing in Angstrom at which a candidate edge is sampled for the occlusion
        test. Default 2.0.

    disorder_threshold : float or None, optional
        Passed through to ``metapredict.predict_disorder_domains``. Default None
        (metapredict default).

    rescue_packed_idrs : bool, optional
        If True (default), stretches that metapredict predicts to be disordered but
        that pack against a predicted folded domain in the structure are treated as
        folded (see :meth:`_rescue_packed_idr_residues`). metapredict sees only the
        sequence, so without this a chain that folds on binding its partner is called
        disordered, gets no surface, and is left out of the SASA calculation, which
        makes the interface on the partner look exposed. Set to False to use the
        metapredict decomposition unchanged.

    Attributes
    ----------
    residues : list of Residue
        All residues across all chains, in chain/sequence order.
    traj : mdtraj.Trajectory
        The loaded structure.
    frontend : object or None
        The FINCHES frontend passed in (None if an ``InteractionMatrixConstructor``
        was passed directly).
    IMC_object : finches.epsilon_calculation.InteractionMatrixConstructor
        The interaction-matrix constructor used internally for all scoring.
    idr_segments : list of dict
        Contiguous IDR runs and their folded-domain anchors (see
        :meth:`_find_idr_segments`).

    Examples
    --------
    >>> from finches.utils.structure_surface import StructureSurface
    >>> from finches.frontend.mpipi_frontend import Mpipi_frontend
    >>> ss = StructureSurface("structure.pdb", Mpipi_frontend())
    >>> scores = ss.surface_vs_idr("RRGRRGRRG")
    """

    def __init__(
        self,
        pdbfilename,
        frontend,
        sequences=None,
        probe_radius=1.4,
        surface_thresh=0.10,
        sasa_mode="v1",
        net_distance_thresh=9.0,
        occlusion_radius=5.0,
        occlusion_surround_thresh=0.3,
        occlusion_sample_spacing=2.0,
        disorder_threshold=None,
        rescue_packed_idrs=True,
    ):
        """
        Load a structure and run the decomposition / surface-classification pipeline.

        On construction the structure is loaded, every protein residue is recorded,
        each chain is decomposed into folded domains and IDRs, and folded-domain
        surface residues are identified. The contiguous-surface net is built lazily on
        first access (see :attr:`surface_graph`). The supplied ``frontend`` provides
        the interaction energetics for all scoring methods. All constructor parameters
        are documented in the class docstring.

        Raises
        ------
        ValueError
            If ``sasa_mode`` is not ``'v1'`` or ``'v2'``, or if ``sequences`` is
            inconsistent with the structure (wrong count, out-of-range ``resSeq``, or
            residue-identity mismatch).
        TypeError
            If ``frontend`` is not a FINCHES frontend (or interaction-matrix
            constructor), or if ``sequences`` is not one of None, str, list/tuple or
            dict.
        """
        if sasa_mode not in ("v1", "v2"):
            raise ValueError("sasa_mode must be 'v1' or 'v2'")

        # accept a frontend (use its IMC_object) or an IMC constructor directly
        if hasattr(frontend, "IMC_object"):
            self.frontend = frontend
            self.IMC_object = frontend.IMC_object
        elif hasattr(frontend, "lookup") and hasattr(
            frontend, "null_interaction_baseline"
        ):
            self.frontend = None
            self.IMC_object = frontend
        else:
            raise TypeError(
                "frontend must be a FINCHES frontend object (e.g. Mpipi_frontend() "
                "or CALVADOS_frontend()) exposing an .IMC_object attribute."
            )

        self.pdbfilename = pdbfilename
        self.probe_radius = probe_radius
        self.surface_thresh = surface_thresh
        self.sasa_mode = sasa_mode
        self.net_distance_thresh = net_distance_thresh
        self.occlusion_radius = occlusion_radius
        self.occlusion_surround_thresh = occlusion_surround_thresh
        self.occlusion_sample_spacing = occlusion_sample_spacing
        self.disorder_threshold = disorder_threshold
        self.rescue_packed_idrs = rescue_packed_idrs

        # load structure (PDB or mmCIF/PDBx)
        self.traj = _load_structure(pdbfilename)

        # all coordinates in Angstrom (mdtraj works in nm)
        self._xyz = self.traj.xyz[0] * 10.0

        # build per-residue records and per-chain sequences
        self.residues = []
        self._by_key = {}
        # memoized linear-sequence weighting context, used for residues that have no
        # surface patch (i.e. IDR residues); keyed by chain index
        self._linear_context_cache = {}
        self._normalize_sequences(sequences)
        self._build_residue_table()

        # workflow steps
        self._decompose()
        self._classify_surface()

        # lazily-built surface net
        self._surface_graph = None
        self._surface_neighbours = None
        self._surface_distance = None

    def __repr__(self):
        """
        Return a one-line summary of the decomposition.

        Returns
        -------
        str
            The file name and the residue counts by class.
        """
        return (
            f"StructureSurface({os.path.basename(str(self.pdbfilename))!r}: "
            f"{len(self._chain_sequences)} chain(s), {len(self.residues)} residues, "
            f"{len(self.folded_residues)} folded of which "
            f"{len(self.surface_residues)} surface, {len(self.idr_residues)} IDR)"
        )

    # ------------------------------------------------------------------
    # sequence handling
    # ------------------------------------------------------------------
    def _normalize_sequences(self, sequences):
        """
        Normalize the ``sequences`` constructor argument into a per-chain dict.

        The result is stored on ``self._provided_sequences`` as a mapping from 0-based
        protein-chain index to the full sequence for that chain (empty when no
        sequences were supplied).

        Parameters
        ----------
        sequences : dict, list, str or None
            See the class docstring for the accepted forms.

        Raises
        ------
        ValueError
            If a single string is given for a multi-chain structure, or a list does
            not have one entry per protein chain.
        TypeError
            If ``sequences`` is not None, str, list/tuple or dict.
        """
        protein_chain_indices = [
            c.index
            for c in self.traj.topology.chains
            if any(r.is_protein for r in c.residues)
        ]
        self._provided_sequences = {}

        if sequences is None:
            return

        if isinstance(sequences, str):
            if len(protein_chain_indices) != 1:
                raise ValueError(
                    "A single sequence string was provided but the structure has "
                    f"{len(protein_chain_indices)} protein chains; pass a dict or list."
                )
            self._provided_sequences[protein_chain_indices[0]] = sequences
        elif isinstance(sequences, dict):
            self._provided_sequences = dict(sequences)
        elif isinstance(sequences, (list, tuple)):
            if len(sequences) != len(protein_chain_indices):
                raise ValueError(
                    f"{len(sequences)} sequences provided for "
                    f"{len(protein_chain_indices)} protein chains."
                )
            self._provided_sequences = dict(zip(protein_chain_indices, sequences))
        else:
            raise TypeError("sequences must be None, str, list/tuple or dict")

    def _chain_full_sequence(self, chain):
        """
        Determine the full sequence and modeled-residue records for one chain.

        If a sequence was provided for the chain it is validated against the modeled
        residues (aligned by ``resSeq``, assumed 1-based into the sequence). Otherwise
        the sequence is reconstructed from the modeled residues, with any ``resSeq``
        numbering gaps filled by placeholder glycines, which become missing (IDR)
        residues.

        Terminal caps (ACE, NME and the like) are skipped: mdtraj flags them as
        protein but they have no alpha carbon and are not residues in the sequence
        sense. A residue whose name has no standard amino acid equivalent (see
        :func:`finches.utils.folded_domain_utils.one_letter_code`) is kept as a
        glycine placeholder with a warning, and is exempt from the identity check
        against a provided sequence.

        Parameters
        ----------
        chain : mdtraj.core.topology.Chain
            The chain to process.

        Returns
        -------
        full_sequence : str
            The complete chain sequence (placeholder ``'G'`` for unknown/missing
            positions when no sequence was provided).
        modeled_records : list of tuple
            ``(res_seq, one_letter, md_index)`` for each residue with coordinates.
        map_resseq : callable
            Function mapping an author ``resSeq`` to its 0-based index in
            ``full_sequence``.
        map_index : callable
            The inverse of ``map_resseq``: maps a 0-based index in ``full_sequence``
            to the author ``resSeq`` a residue at that position carries, or would
            carry if it is missing from the coordinates.

        Raises
        ------
        ValueError
            If a provided sequence is too short for a modeled ``resSeq`` or a residue
            identity does not match the provided sequence.
        """
        modeled = []
        unknown = set()
        for r in chain.residues:
            if not r.is_protein:
                continue
            if not any(a.name == "CA" for a in r.atoms):
                continue
            aa = one_letter_code(r)
            if aa is None:
                unknown.add(r.index)
                aa = "G"
            modeled.append((r.resSeq, aa, r.index))

        if unknown:
            names = sorted({self.traj.topology.residue(i).name for i in unknown})
            warnings.warn(
                f"Chain {chain.index}: {len(unknown)} residue(s) named {names} have "
                "no standard amino acid equivalent and are treated as glycine "
                "placeholders."
            )

        provided = self._provided_sequences.get(chain.index)
        if provided is not None:
            for res_seq, aa, md_index in modeled:
                pos = res_seq - 1
                if pos < 0 or pos >= len(provided):
                    raise ValueError(
                        f"Chain {chain.index}: residue resSeq={res_seq} falls "
                        f"outside the provided sequence (length {len(provided)}). "
                        "resSeq is assumed to be 1-based into the sequence."
                    )
                if md_index not in unknown and provided[pos] != aa:
                    raise ValueError(
                        f"Chain {chain.index}: residue identity mismatch at "
                        f"resSeq={res_seq}: structure has '{aa}' but the provided "
                        f"sequence has '{provided[pos]}'."
                    )
            return provided, modeled, (lambda rs: rs - 1), (lambda idx: idx + 1)

        if not modeled:
            return "", modeled, (lambda rs: rs), (lambda idx: idx + 1)

        # reconstruct from modeled residues, filling resSeq gaps as missing (IDR)
        # residues that keep the author number of the gap they fill
        min_rs = min(m[0] for m in modeled)
        max_rs = max(m[0] for m in modeled)
        seqlist = ["G"] * (max_rs - min_rs + 1)
        for res_seq, aa, _ in modeled:
            seqlist[res_seq - min_rs] = aa
        full = "".join(seqlist)
        return full, modeled, (lambda rs: rs - min_rs), (lambda idx: idx + min_rs)

    # ------------------------------------------------------------------
    # residue table + decomposition
    # ------------------------------------------------------------------
    def _build_residue_table(self):
        """
        Build the per-residue :class:`Residue` records for every protein chain.

        Populates ``self.residues`` (in chain/sequence order), ``self._by_key`` (keyed
        by ``(chain_index, res_seq)``), ``self._chain_sequences`` (full sequence per
        chain) and ``self._chain_resseq_to_index``. Residues present in the
        coordinates are flagged ``modeled=True`` and carry their mdtraj residue index;
        residues that are only in the full sequence are recorded as missing.

        Returns
        -------
        None
        """
        self._chain_sequences = {}
        self._chain_resseq_to_index = {}

        for chain in self.traj.topology.chains:
            if not any(r.is_protein for r in chain.residues):
                continue

            full, modeled, map_resseq, map_index = self._chain_full_sequence(chain)
            self._chain_sequences[chain.index] = full

            modeled_by_seqidx = {}
            for res_seq, aa, md_index in modeled:
                modeled_by_seqidx[map_resseq(res_seq)] = (res_seq, aa, md_index)

            resseq_to_index = {}
            for seq_index, aa in enumerate(full):
                if seq_index in modeled_by_seqidx:
                    res_seq, true_aa, md_index = modeled_by_seqidx[seq_index]
                    rec = Residue(chain.index, res_seq, seq_index, true_aa, True)
                    rec.md_index = md_index
                else:
                    # missing residue: it carries the author number it would have had,
                    # so its key can never collide with a modeled residue's key
                    res_seq = map_index(seq_index)
                    rec = Residue(chain.index, res_seq, seq_index, aa, False)

                self.residues.append(rec)
                self._by_key[rec.key] = rec
                resseq_to_index[rec.res_seq] = seq_index

            self._chain_resseq_to_index[chain.index] = resseq_to_index

    def _decompose(self):
        """
        Assign every residue to a folded domain or an IDR (capability 1).

        Each chain's full sequence is passed to
        ``metapredict.predict_disorder_domains``; a residue is labelled ``'folded'``
        only if it is both modeled and inside a predicted folded-domain boundary, and
        ``'idr'`` otherwise (so all missing residues are IDR). If metapredict fails for
        a chain, its modeled region is treated as entirely folded. If
        ``rescue_packed_idrs`` is set, predicted-IDR stretches that pack against a
        folded domain in the structure are then relabelled folded. The contiguous IDR
        runs and their anchors are finally recorded on ``self.idr_segments``.

        Returns
        -------
        None
        """
        for chain_index, full in self._chain_sequences.items():
            if len(full) == 0:
                continue

            folded_positions = set()
            try:
                dd = meta.predict_disorder_domains(
                    full, disorder_threshold=self.disorder_threshold
                )
                for start, end in dd.folded_domain_boundaries:
                    folded_positions.update(range(start, end))
            except Exception as e:
                # if metapredict fails, treat the whole modeled region as folded,
                # but say so: silently losing the IDRs is worse than a warning
                warnings.warn(
                    f"metapredict failed on chain {chain_index} ({e!r}); treating "
                    "every modeled residue in the chain as folded."
                )
                folded_positions = set(range(len(full)))

            for rec in self._chain_residues(chain_index):
                # a residue is part of a folded domain only if it is BOTH modeled
                # and predicted folded; missing residues are always IDR
                if rec.modeled and rec.seq_index in folded_positions:
                    rec.domain = "folded"
                else:
                    rec.domain = "idr"

        if self.rescue_packed_idrs:
            self._rescue_packed_idr_residues()

        # record IDR segments (contiguous IDR runs) with their folded anchors
        self.idr_segments = self._find_idr_segments()

    def _folded_partner_counts(self):
        """
        Count, for every modeled IDR residue, the folded-domain residues it touches.

        A partner is a residue of a predicted folded domain, in any chain, with a heavy
        atom within ``PACKING_CONTACT_CUTOFF`` Angstrom of one of the IDR residue's
        heavy atoms. Partners on the same chain fewer than
        ``PACKING_MIN_SEQUENCE_SEPARATION`` positions away in sequence are ignored, so
        a residue next to a domain boundary does not count as packed just because it
        is bonded to the domain.

        Returns
        -------
        dict
            Maps the ``md_index`` of each modeled IDR residue to its number of distinct
            folded-domain partners. Residues with no partner are absent.
        """
        records_by_md = {r.md_index: r for r in self.residues if r.modeled}
        topology = self.traj.topology

        folded_atoms = []
        folded_atom_residue = []
        idr_atoms = []
        idr_atom_residue = []
        for atom in topology.atoms:
            record = records_by_md.get(atom.residue.index)
            if record is None or atom.element.symbol == "H":
                continue
            if record.domain == "folded":
                folded_atoms.append(atom.index)
                folded_atom_residue.append(record)
            else:
                idr_atoms.append(atom.index)
                idr_atom_residue.append(record)
        if not folded_atoms or not idr_atoms:
            return {}

        folded_tree = cKDTree(self._xyz[np.array(folded_atoms)])
        hits_per_atom = folded_tree.query_ball_point(
            self._xyz[np.array(idr_atoms)], r=PACKING_CONTACT_CUTOFF
        )

        partners = {}
        for idr_record, hits in zip(idr_atom_residue, hits_per_atom):
            for hit in hits:
                folded_record = folded_atom_residue[hit]
                is_local = (
                    folded_record.chain_index == idr_record.chain_index
                    and abs(folded_record.seq_index - idr_record.seq_index)
                    < PACKING_MIN_SEQUENCE_SEPARATION
                )
                if is_local:
                    continue
                partners.setdefault(idr_record.md_index, set()).add(
                    folded_record.md_index
                )
        return {md_index: len(found) for md_index, found in partners.items()}

    def _rescue_packed_idr_residues(self):
        """
        Treat predicted-IDR stretches that pack against a folded domain as folded.

        metapredict works from sequence alone, so a segment that only folds in context
        - a helix packed against a partner chain, or the edge of a domain that a
        predicted boundary cuts off - is called disordered even though the structure
        shows it packed. Left as IDR, such a segment gets no surface, and because IDR
        atoms are removed before the SASA calculation, the interface it covers on the
        folded domain looks exposed.

        Each modeled residue of a predicted IDR run is scored by its number of
        folded-domain partners (see :meth:`_folded_partner_counts`). A residue is packed
        when that count, averaged over a ``PACKING_WINDOW``-residue window centred on
        it within the run, is at least ``PACKING_MIN_PARTNERS``. Gaps shorter than the
        window between packed residues, or between a packed residue and the folded
        domain flanking the run, are closed; isolated packed stretches shorter than
        ``PACKING_MIN_RUN`` are dropped. Packed residues get ``domain='folded'`` and
        ``packed_rescue=True``.

        Contacts are counted only against residues that metapredict already calls
        folded, so a structure in which every residue is predicted disordered is left
        unchanged, and two IDRs packed only against each other are not rescued.

        Returns
        -------
        None
        """
        partner_counts = self._folded_partner_counts()
        if not partner_counts:
            return

        for segment in self._find_idr_segments():
            run = segment["residues"]
            modeled = np.array([r.modeled for r in run])
            # missing residues have md_index None and so no partners
            counts = np.array(
                [partner_counts.get(r.md_index, 0) for r in run], dtype=float
            )
            packed = modeled & (
                _windowed_mean(counts, PACKING_WINDOW) >= PACKING_MIN_PARTNERS
            )

            n_flank_folded = segment["n_anchor"] is not None
            c_flank_folded = segment["c_anchor"] is not None
            packed = _fill_short_gaps(
                packed, modeled, n_flank_folded, c_flank_folded, PACKING_WINDOW
            )
            packed = _drop_short_isolated_runs(
                packed, n_flank_folded, c_flank_folded, PACKING_MIN_RUN
            )

            for rec, is_packed in zip(run, packed):
                if is_packed:
                    rec.domain = "folded"
                    rec.packed_rescue = True

    def _find_idr_segments(self):
        """
        Identify contiguous IDR runs per chain and attach folded-domain anchors.

        An anchor is the modeled folded residue immediately flanking an IDR run; it is
        the physical attachment point from which the polymer-reach model is measured.
        Terminal IDRs have a single anchor; internal loops can have two.

        Returns
        -------
        list of dict
            One dict per IDR run with keys ``chain_index`` (int), ``residues``
            (ordered list of :class:`Residue`), ``n_anchor`` and ``c_anchor`` (the
            flanking modeled folded :class:`Residue`, or None).
        """
        segments = []
        for chain_index in self._chain_sequences:
            chain_recs = self._chain_residues(chain_index)
            i = 0
            n = len(chain_recs)
            while i < n:
                if chain_recs[i].domain != "idr":
                    i += 1
                    continue
                j = i
                while j < n and chain_recs[j].domain == "idr":
                    j += 1
                seg_recs = chain_recs[i:j]
                n_anchor = chain_recs[i - 1] if i > 0 else None
                c_anchor = chain_recs[j] if j < n else None
                # only modeled folded residues can act as physical anchors
                if n_anchor is not None and not (
                    n_anchor.modeled and n_anchor.domain == "folded"
                ):
                    n_anchor = None
                if c_anchor is not None and not (
                    c_anchor.modeled and c_anchor.domain == "folded"
                ):
                    c_anchor = None
                segments.append(
                    {
                        "chain_index": chain_index,
                        "residues": seg_recs,
                        "n_anchor": n_anchor,
                        "c_anchor": c_anchor,
                    }
                )
                i = j
        return segments

    def _chain_residues(self, chain_index):
        """
        Return the residue records for a chain in sequence order.

        Parameters
        ----------
        chain_index : int
            0-based mdtraj chain index.

        Returns
        -------
        list of Residue
            The chain's residues, ordered by sequence position.
        """
        return [r for r in self.residues if r.chain_index == chain_index]

    # ------------------------------------------------------------------
    # surface classification (capability 2)
    # ------------------------------------------------------------------
    def _classify_surface(self):
        """
        Identify solvent-exposed folded-domain residues (capability 2).

        SASA is computed with ``mdtraj.shrake_rupley`` on a sub-structure containing
        only folded-domain atoms, so dangling IDR tails cannot artificially occlude
        the folded-domain surface. A residue is flagged ``surface=True`` when its SASA
        exceeds ``surface_thresh`` times its maximum reference SASA (side-chain only
        for ``sasa_mode='v1'``, side-chain + backbone for ``'v2'``). Side-chain
        centre-of-mass positions (Angstrom) are stored for surface residues, and the
        folded-domain heavy-atom cloud is cached for the occlusion test.

        Returns
        -------
        None
        """
        folded = [r for r in self.residues if r.domain == "folded" and r.modeled]
        if not folded:
            self._folded_heavy_xyz = np.zeros((0, 3))
            return

        # process folded residues in ascending mdtraj index order so they line up
        # with the residue order produced by atom_slice + shrake_rupley below
        folded = sorted(folded, key=lambda r: r.md_index)

        # atom indices belonging to folded residues
        fd_atom_indices = []
        for rec in folded:
            res = self.traj.topology.residue(rec.md_index)
            fd_atom_indices.extend(a.index for a in res.atoms)
        fd_atom_indices = np.array(sorted(fd_atom_indices))

        # SASA on the folded-domain-only sub-structure (IDR atoms removed so they
        # cannot occlude the folded-domain surface)
        fd_traj = self.traj.atom_slice(fd_atom_indices)
        sasa = (
            100.0
            * md.shrake_rupley(
                fd_traj, mode="residue", probe_radius=self.probe_radius * 0.1
            )[0]
        )

        # fd_traj residues are in the same (ascending-index) order as `folded`
        for rec, residue_sasa in zip(folded, sasa):
            max_sc, max_bb = MAX_SASA_DATA.get(rec.one_letter, (1.0, 1.0))
            if self.sasa_mode == "v1":
                threshold = self.surface_thresh * max_sc
            else:
                threshold = self.surface_thresh * (max_sc + max_bb)
            rec.surface = bool(residue_sasa > threshold)

        # side-chain centre-of-mass positions (Angstrom) for surface residues
        for rec in folded:
            if not rec.surface:
                continue
            rec.position = self._sidechain_com(rec.md_index)

        # heavy-atom cloud of the folded domain(s) for the occlusion test
        heavy = [
            a.index
            for idx in (r.md_index for r in folded)
            for a in self.traj.topology.residue(idx).atoms
            if a.element.symbol != "H"
        ]
        self._folded_heavy_xyz = (
            self._xyz[np.array(heavy)] if heavy else np.zeros((0, 3))
        )

    def _sidechain_com(self, md_index):
        """
        Compute the side-chain centre of mass of a residue.

        Falls back to the C-alpha atom (e.g. for glycine) and then to all atoms if no
        side-chain atoms are present.

        Parameters
        ----------
        md_index : int
            Global mdtraj residue index.

        Returns
        -------
        numpy.ndarray
            The centre-of-mass coordinate in Angstrom, shape ``(3,)``.
        """
        top = self.traj.topology
        atom_indices = top.select(f"sidechain and resid {md_index}")
        if len(atom_indices) == 0:
            atom_indices = top.select(f"resid {md_index} and name CA")
        if len(atom_indices) == 0:
            atom_indices = np.array([a.index for a in top.residue(md_index).atoms])
        com_nm = md.compute_center_of_mass(self.traj.atom_slice(atom_indices))[0]
        return com_nm * 10.0

    # ------------------------------------------------------------------
    # public accessors for the decomposition
    # ------------------------------------------------------------------
    @property
    def folded_residues(self):
        """
        Folded-domain residues that are present in the coordinates.

        Returns
        -------
        list of Residue
            Modeled residues assigned to a folded domain.
        """
        return [r for r in self.residues if r.domain == "folded" and r.modeled]

    @property
    def surface_residues(self):
        """
        Solvent-exposed folded-domain residues.

        Returns
        -------
        list of Residue
            Modeled folded residues flagged as surface-exposed.
        """
        return [r for r in self.residues if r.surface]

    @property
    def idr_residues(self):
        """
        IDR residues, including residues missing from the coordinates.

        Returns
        -------
        list of Residue
            All residues assigned to the ``'idr'`` domain.
        """
        return [r for r in self.residues if r.domain == "idr"]

    def get_residue(self, chain_index, res_seq):
        """
        Look up a residue record by chain index and author residue number.

        Parameters
        ----------
        chain_index : int
            0-based mdtraj chain index.
        res_seq : int
            Author residue number (PDB ``resSeq``).

        Returns
        -------
        Residue
            The matching residue record.

        Raises
        ------
        KeyError
            If no residue with the given key exists.
        """
        return self._by_key[(chain_index, res_seq)]

    # ------------------------------------------------------------------
    # contiguous-surface net (capability 3)
    # ------------------------------------------------------------------
    def _build_surface_net(self):
        """
        Build the contiguous-surface neighbour graph over surface residues (cap. 3).

        Candidate neighbour pairs are found with a KD-tree on the surface residues'
        side-chain centres-of-mass (within ``net_distance_thresh``); each candidate
        edge is kept only if it passes the occlusion test in
        :meth:`_segment_on_surface` (i.e. it hugs the surface rather than cutting
        through the buried core). Populates ``self._surface_graph`` (a weighted
        ``networkx.Graph``), ``self._surface_neighbours`` and the geodesic
        ``self._surface_distance``. Because all chains contribute points, neighbours
        may span chains.

        Returns
        -------
        None
        """
        surf = self.surface_residues
        graph = nx.Graph()
        for rec in surf:
            graph.add_node(rec.key)

        if len(surf) < 2:
            self._surface_graph = graph
            self._surface_neighbours = {r.key: [] for r in surf}
            self._surface_distance = dict(nx.all_pairs_dijkstra_path_length(graph))
            return

        positions = np.array([r.position for r in surf])
        keys = [r.key for r in surf]
        point_tree = cKDTree(positions)
        atom_tree = (
            cKDTree(self._folded_heavy_xyz) if len(self._folded_heavy_xyz) else None
        )

        for a, b in point_tree.query_pairs(r=self.net_distance_thresh):
            p1, p2 = positions[a], positions[b]
            if atom_tree is None or self._segment_on_surface(p1, p2, atom_tree):
                graph.add_edge(keys[a], keys[b], weight=float(np.linalg.norm(p1 - p2)))

        self._surface_graph = graph
        self._surface_neighbours = {k: list(graph.neighbors(k)) for k in graph.nodes}
        self._surface_distance = dict(
            nx.all_pairs_dijkstra_path_length(graph, weight="weight")
        )

    def _segment_on_surface(self, p1, p2, atom_tree):
        """
        Test whether the segment between two surface points hugs the surface.

        The segment is sampled at interior points spaced by
        ``occlusion_sample_spacing``. At each sample the unit vectors to heavy atoms
        within ``occlusion_radius`` are averaged; a small resultant magnitude means
        atoms surround the point on all sides (the point is buried). Any buried sample
        means the segment cuts through the interior and the edge is rejected.

        Parameters
        ----------
        p1, p2 : numpy.ndarray
            Endpoint coordinates in Angstrom, shape ``(3,)``.
        atom_tree : scipy.spatial.cKDTree
            KD-tree built over the folded-domain heavy-atom coordinates.

        Returns
        -------
        bool
            True if the segment stays on the surface, False if it passes through the
            buried core.
        """
        length = np.linalg.norm(p2 - p1)
        n_samples = max(int(length / self.occlusion_sample_spacing), 1)
        # interior sample fractions only (endpoints are surface COMs by definition)
        for t in np.linspace(0.0, 1.0, n_samples + 2)[1:-1]:
            point = p1 + t * (p2 - p1)
            idxs = atom_tree.query_ball_point(point, self.occlusion_radius)
            if len(idxs) < 4:
                continue  # sparse neighbourhood -> exposed
            vecs = self._folded_heavy_xyz[idxs] - point
            norms = np.linalg.norm(vecs, axis=1)
            nz = norms > 1e-6
            if not np.any(nz):
                continue
            units = vecs[nz] / norms[nz, None]
            resultant = np.linalg.norm(units.mean(axis=0))
            if resultant < self.occlusion_surround_thresh:
                return False
        return True

    @property
    def surface_graph(self):
        """
        The contiguous-surface neighbour graph (built lazily on first access).

        Returns
        -------
        networkx.Graph
            Nodes are surface-residue keys ``(chain_index, res_seq)``; edges connect
            contiguous-surface neighbours and carry a ``weight`` equal to the
            Euclidean distance (Angstrom) between side-chain centres-of-mass.
        """
        if self._surface_graph is None:
            self._build_surface_net()
        return self._surface_graph

    @property
    def surface_neighbours(self):
        """
        Per-residue contiguous-surface neighbours (built lazily on first access).

        Returns
        -------
        dict
            Maps each surface residue key to a list of its neighbour keys.
        """
        if self._surface_neighbours is None:
            self._build_surface_net()
        return self._surface_neighbours

    @property
    def surface_distance(self):
        """
        Geodesic (over-surface) distances between surface residues.

        Returns
        -------
        dict
            Nested mapping ``{key_a: {key_b: distance}}`` of shortest-path distances
            (Angstrom) along the surface net (``networkx`` Dijkstra path lengths).
        """
        if self._surface_distance is None:
            self._build_surface_net()
        return self._surface_distance

    def surface_patch(self, key, include_self=True):
        """
        Return the surface-patch residue keys for a surface residue.

        The patch is the local surface neighbourhood used as the "context" for the
        FINCHES-style charge/aliphatic weighting (capability 4).

        Parameters
        ----------
        key : tuple
            ``(chain_index, res_seq)`` of a surface residue.
        include_self : bool, optional
            Whether to include ``key`` itself. Default True.

        Returns
        -------
        list of tuple
            Residue keys forming the local surface patch (the residue and its
            contiguous-surface neighbours).
        """
        patch = list(self.surface_neighbours.get(key, []))
        if include_self:
            patch = [key] + patch
        return patch

    # ------------------------------------------------------------------
    # context-aware surface interaction (capability 4)
    # ------------------------------------------------------------------
    def _patch_charge_counts(self, key):
        """
        Count positively and negatively charged residues in a surface patch.

        Parameters
        ----------
        key : tuple
            ``(chain_index, res_seq)`` of a surface residue.

        Returns
        -------
        tuple of (int, int, int)
            ``(n_pos, n_neg, patch_len)`` for the residue's surface patch (the residue
            plus its contiguous-surface neighbours).
        """
        patch = self.surface_patch(key)
        n_pos = sum(1 for k in patch if self._by_key[k].one_letter in POSITIVE_RESIDUES)
        n_neg = sum(1 for k in patch if self._by_key[k].one_letter in NEGATIVE_RESIDUES)
        return n_pos, n_neg, len(patch)

    def _patch_aliphatic_level(self, key):
        """
        Aliphatic clustering level of a surface residue's patch.

        Mirrors ``finches.parsing_aminoacid_sequences.get_aliphatic_groups`` but over
        the surface neighbourhood instead of the linear sequence: a non-aliphatic
        centre is 0, and an aliphatic centre is the count of aliphatic residues in its
        patch (including itself), capped at 3.

        Parameters
        ----------
        key : tuple
            ``(chain_index, res_seq)`` of a surface residue.

        Returns
        -------
        int
            The clustering level (0, 1, 2 or 3).
        """
        center = self._by_key[key]
        if center.one_letter not in ALIPHATIC_RESIDUES:
            return 0
        # count of aliphatic residues in the patch (including self), capped at 3
        count = sum(
            1
            for k in self.surface_patch(key)
            if self._by_key[k].one_letter in ALIPHATIC_RESIDUES
        )
        return min(count, 3)

    def _chain_linear_context(self, chain_index):
        """
        Per-position linear weighting context for a whole chain.

        Residues with no contiguous-surface patch (IDR residues) take their charge and
        aliphatic context from the linear sequence instead, exactly as FINCHES does for
        a disordered chain. Computed once per chain and memoized.

        Parameters
        ----------
        chain_index : int
            Index of the chain.

        Returns
        -------
        tuple of numpy.ndarray
            ``(pos_window, neg_window, aliphatic_levels)`` over the chain's full
            sequence.
        """
        if chain_index not in self._linear_context_cache:
            seq = self._chain_sequences[chain_index]
            pos_w, neg_w = get_charge_window_counts(seq)
            self._linear_context_cache[chain_index] = (
                pos_w,
                neg_w,
                np.array(get_aliphatic_groups(seq)),
            )

        return self._linear_context_cache[chain_index]

    def _residue_context(self, rec):
        """
        Charge and aliphatic weighting context for one residue.

        A folded-domain surface residue takes its context from its contiguous-surface
        patch, i.e. the residues it is actually adjacent to in 3D. Any other residue
        (an IDR residue, which has no surface patch) falls back to the linear +/-1
        sequence window and sequence aliphatic clustering used by FINCHES.

        Parameters
        ----------
        rec : Residue
            The residue to describe.

        Returns
        -------
        tuple of (float, float, int)
            ``(n_pos, n_neg, aliphatic_level)`` for the residue's local environment.
        """
        if rec.surface:
            n_pos, n_neg, _ = self._patch_charge_counts(rec.key)
            return n_pos, n_neg, self._patch_aliphatic_level(rec.key)

        pos_w, neg_w, ali = self._chain_linear_context(rec.chain_index)
        i = rec.seq_index

        return float(pos_w[i]), float(neg_w[i]), int(ali[i])

    def _idr_context(self, idr_sequence):
        """
        Everything the weighting needs to know about an IDR, computed once.

        The IDR takes its charge and aliphatic context from the linear sequence,
        exactly as it would in an IDR:IDR calculation, using the same helpers as
        FINCHES (``finches.parsing_aminoacid_sequences``). The context is evaluated
        over the full IDR, so residues at the edge of a sliding window keep their
        true sequence neighbours.

        Parameters
        ----------
        idr_sequence : str
            The IDR amino acid sequence.

        Returns
        -------
        dict
            ``seq`` (the sequence after the IMC's sequence converter), ``pos`` and
            ``neg`` (per-position counts of positive and negative residues in the +/-1
            window), ``ali`` (per-position aliphatic clustering levels) and
            ``charged`` (boolean array, True where the residue is charged). All
            arrays have length ``len(idr_sequence)``.
        """
        seq = self.IMC_object.sequence_converter(idr_sequence)
        pos, neg = get_charge_window_counts(seq)
        return {
            "seq": seq,
            "pos": pos,
            "neg": neg,
            "ali": np.array(get_aliphatic_groups(seq)),
            "charged": np.array([r in CHARGED_RESIDUES for r in seq], dtype=bool),
        }

    def anchor_position(self, anchor):
        """
        Return the 3D position used to represent an anchoring residue.

        Parameters
        ----------
        anchor : tuple
            ``(chain_index, res_seq)`` of the anchoring residue.

        Returns
        -------
        numpy.ndarray
            The residue's position, falling back to its side-chain centre of mass
            when no position has been assigned.
        """
        rec = self._by_key[anchor]
        if rec.position is None:
            return self._sidechain_com(rec.md_index)
        return rec.position

    @staticmethod
    def _contour_separation(idr_length, tether):
        """
        Contour separation of each IDR residue from its tether point.

        The reach model is indexed by how many residues along the chain a given IDR
        position sits from the point at which the IDR is attached to the structure, so
        which terminus is tethered determines the direction of counting.

        Parameters
        ----------
        idr_length : int
            Number of residues in the IDR.

        tether : str
            ``'N'`` if the IDR's N-terminal residue is attached to the anchor (the IDR
            runs C-terminally away from the folded domain), or ``'C'`` if its
            C-terminal residue is attached (the IDR runs N-terminally away).

        Returns
        -------
        numpy.ndarray
            1-based contour separation for each IDR position, length ``idr_length``.

        Raises
        ------
        ValueError
            If ``tether`` is not ``'N'`` or ``'C'``.
        """
        idx = np.arange(1, idr_length + 1)

        if tether == "N":
            return idx
        if tether == "C":
            return idx[::-1].copy()

        raise ValueError(f"tether must be 'N' or 'C', got {tether!r}")

    def idr_tether(self, segment):
        """
        Return the anchor and tethered terminus for an IDR segment.

        Getting the tethered terminus the wrong way round silently inverts the reach
        model, so this resolves it from the segment's own anchors rather than leaving
        it to the caller. A segment anchored at its C-terminal end (i.e. the IDR runs
        off the N-terminal side of a folded domain) is tethered by the IDR's
        C-terminus, and vice versa.

        Parameters
        ----------
        segment : dict or int
            An entry from :attr:`idr_segments`, or an index into it.

        Returns
        -------
        tuple
            ``(anchor_key, tether)`` where ``anchor_key`` is a
            ``(chain_index, res_seq)`` tuple and ``tether`` is ``'N'`` or ``'C'``,
            suitable for passing straight to :meth:`surface_vs_idr_matrix`.

        Raises
        ------
        ValueError
            If the segment has no folded-domain anchor at either end.
        """
        if isinstance(segment, (int, np.integer)):
            segment = self.idr_segments[segment]

        if segment.get("c_anchor") is not None:
            return segment["c_anchor"].key, "C"

        if segment.get("n_anchor") is not None:
            return segment["n_anchor"].key, "N"

        raise ValueError(
            "IDR segment has no folded-domain anchor at either terminus, so it "
            "cannot be tethered for a cis calculation"
        )

    def footprint_order(self, key):
        """
        Surface residues ordered by geodesic distance from a centre residue.

        The over-the-surface (Dijkstra) distance is used rather than a straight line,
        because an IDR reaching across the domain has to travel over the surface rather
        than through the protein. Residues in a disconnected surface component are
        unreachable and are omitted.

        Parameters
        ----------
        key : tuple
            ``(chain_index, res_seq)`` of the centre residue.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            ``(order, distances)`` where ``order`` indexes into
            :attr:`surface_residues` sorted by increasing geodesic distance from
            ``key``, and ``distances`` are the matching distances in Angstrom.
        """
        dists = self.surface_distance.get(key, {})
        idx, d = [], []
        for n, rec in enumerate(self.surface_residues):
            if rec.key in dists:
                idx.append(n)
                d.append(dists[rec.key])

        idx = np.asarray(idx, dtype=int)
        d = np.asarray(d, dtype=float)
        order = np.argsort(d, kind="stable")

        return idx[order], d[order]

    def surface_vs_idr_matrix(
        self,
        idr_sequence,
        window_size=31,
        anchor=None,
        tether=None,
        rows="surface",
        use_footprint=True,
        reach_b=DEFAULT_REACH_B,
        reach_nu=DEFAULT_REACH_NU,
        contact_radius=DEFAULT_CONTACT_RADIUS,
        use_charge_weighting=True,
        use_aliphatic_weighting=True,
    ):
        """
        Build a surface:IDR interaction map over sliding windows of an IDR.

        Every surface residue (or every residue, see ``rows``) is scored against every
        sliding window of the IDR, giving the structural analogue of a FINCHES IDR:IDR
        intermap.

        **Contact footprint.** A window of ``window_size`` residues laid over the surface
        does not touch one residue, it touches a patch of them, and how many depends on
        how long the window is. The window's centre residue is treated as pinned over the
        row's residue; the residue ``k`` positions along the chain is then typically
        ``R(|k|) = reach_b * |k|**reach_nu`` Angstrom away, and can contact surface
        residues at that sort of geodesic (over-the-surface) distance from the pinned
        point. The centre of the footprint is therefore seen by every residue in the
        window, while distant surface is seen only by the residues near the window's
        ends and so contributes proportionately less.

        Two details make this behave like a real contact rather than a point:

        - ``R(|k|)`` is the *mean* end-to-end distance of a ``|k|``-residue segment, not a
          hard limit, so contact weight falls off as the Gaussian
          ``exp(-3 d**2 / (2 R**2))`` rather than switching off at a cutoff. A top-hat
          mask makes a residue flip in and out of the footprint as the pinned point moves
          by an Angstrom, which puts steps into the map that the structure does not have.

        - the reach is floored at ``contact_radius`` via
          ``R_eff = sqrt(contact_radius**2 + R**2)``, because even the pinned residue has
          a side chain and touches a shell of surface rather than a single point.

        Together these mean two surface residues that touch each other see almost the
        same window, which is the physical expectation and what makes neighbouring rows
        of the map agree. The weights are normalised at every ``k``, so this changes how
        the footprint is averaged, not the units.

        **Units.** Each cell is a sum over the IDR window of a mean over the surface
        residues that each IDR residue can reach. That is the same form as a FINCHES
        IDR:IDR intermap cell (a sum over one window of a mean over the other), so values
        are on a comparable scale, and are independent of how many surface neighbours a
        residue happens to have.

        **Local chemistry.** The contiguous-surface patch is still used, but only as the
        chemical context for the charge and aliphatic weighting, exactly as the +/-1
        sequence window and sequence aliphatic clustering are used on the IDR side.

        Parameters
        ----------
        idr_sequence : str
            The IDR amino acid sequence to challenge the surface with.

        window_size : int, optional
            Size of the sliding window over the IDR. Must be odd and no longer than the
            sequence. Default 31, for consistency with FINCHES IDR:IDR intermaps. Note
            this now sets the size of the surface footprint as well as the IDR window.

        anchor : tuple, optional
            ``(chain_index, res_seq)`` of the structure residue the IDR is tethered to.
            Required when ``tether`` is set, ignored otherwise.

        tether : str or None, optional
            ``None`` for a trans calculation (default). ``'N'`` or ``'C'`` to run in cis,
            naming which terminus of ``idr_sequence`` is attached to ``anchor``. In cis
            each IDR position is additionally attenuated by how far it has to stray from
            the anchor to reach the row's residue, on the same Gaussian falloff used for
            the contact footprint. Cells therefore tend towards zero far from the anchor
            rather than being cut off at one, and the trans result is recovered wherever
            the whole domain sits well inside the tether's reach.

        rows : str, optional
            ``'surface'`` (default) gives one row per folded-domain surface residue.
            ``'all'`` gives one row per residue in the structure, in chain and sequence
            order, with buried folded residues filled with ``np.nan`` so they can be
            greyed out. The structure's own IDR residues are scored too: they take
            their weighting context from the linear sequence, and their footprint runs
            along their own disordered segment, with the typical chain distance
            ``R(s)`` of the residue ``s`` positions away standing in for the geodesic
            distance. That keeps IDR rows on the same scale as surface rows. In a cis
            calculation IDR rows are ``np.nan``, because a residue with no fixed
            position has no defined distance from the anchor.

        use_footprint : bool, optional
            If True (default), aggregate over the window-sized contact footprint
            described above. If False, aggregate over the contiguous-surface patch alone
            for every IDR residue, which reduces the cell exactly to
            ``IMC.calculate_epsilon_value(window_sequence, patch_sequence)``. For a row
            belonging to one of the structure's own IDR residues it reduces to the flat
            mean over the ``window_size`` stretch of that segment centred on the
            residue (truncated at the segment ends), i.e. exactly an IDR:IDR intermap
            cell.

        reach_b, reach_nu : float, optional
            Polymer-reach parameters ``R(n) = reach_b * n**reach_nu`` (Angstrom). These
            set both the footprint size and, in cis, the tether constraint.

        contact_radius : float, optional
            Radius in Angstrom of the contact shell of a single residue, used to floor
            both the footprint reach and the cis tether reach as described above. Default
            :data:`DEFAULT_CONTACT_RADIUS`. Larger values give a smoother, more spatially
            averaged map; set to 0 to treat each IDR residue as a point contact. Has no
            effect on the footprint when ``use_footprint`` is False, but still applies to
            the cis tether.

        use_charge_weighting, use_aliphatic_weighting : bool, optional
            Toggle the FINCHES-style weighting terms. Default True.

        Returns
        -------
        tuple
            A 3-element tuple:

            [0] : numpy.ndarray
                The interaction map, shape ``(n_rows, n_windows)`` where
                ``n_windows = len(idr_sequence) - window_size + 1``. Negative is
                attractive, positive repulsive; unscorable rows are ``np.nan``.

            [1] : list of tuple
                The ``(chain_index, res_seq)`` key for each row, in row order.

            [2] : numpy.ndarray
                The 0-based index into ``idr_sequence`` of each window's centre residue.

        Raises
        ------
        ValueError
            If ``window_size`` is even, non-positive, or longer than the sequence; if
            ``tether`` is set without an ``anchor``; or if ``rows`` is not recognised.
        """
        if rows == "surface":
            records = list(self.surface_residues)
        elif rows == "all":
            records = sorted(self.residues, key=lambda r: (r.chain_index, r.seq_index))
        else:
            raise ValueError(f"rows must be 'surface' or 'all', got {rows!r}")

        n_idr = len(idr_sequence)

        if window_size <= 0 or window_size % 2 != 1:
            raise ValueError(
                f"window_size must be a positive odd integer, got {window_size}"
            )

        if n_idr == 0 or len(records) == 0:
            return np.zeros((0, 0)), [], np.zeros(0, dtype=int)

        if window_size > n_idr:
            raise ValueError(
                f"window_size ({window_size}) is longer than the IDR sequence ({n_idr})"
            )

        if tether is not None and anchor is None:
            raise ValueError(
                "a cis calculation (tether set) requires an anchor=(chain, resSeq)"
            )

        if contact_radius < 0:
            raise ValueError(f"contact_radius must be >= 0, got {contact_radius}")

        if tether is not None and tether not in ("N", "C"):
            raise ValueError(f"tether must be 'N', 'C' or None, got {tether!r}")

        baseline = self.IMC_object.null_interaction_baseline
        idr_ctx = self._idr_context(idr_sequence)

        half = window_size // 2
        n_windows = n_idr - window_size + 1
        window_centres = np.arange(n_windows) + half

        # Per-residue deviation from the non-interacting baseline, for every surface
        # residue against every IDR position. Computed once and reused by every
        # footprint, since a residue's own weighted row does not depend on which
        # footprint it is being aggregated into.
        #
        # NOTE the 2 * baseline. FINCHES splits a matrix into attractive and repulsive
        # parts, zeroing the elements that fall on the other side of the baseline, and
        # then subtracts the baseline from *both* matrices before summing them. Every
        # element therefore ends up carrying -2 * baseline rather than -baseline. This
        # has to match for these values to be comparable with an IDR:IDR intermap, and
        # it is what makes a poly(GS)-like contact score zero.
        surf_dev = np.zeros((len(self.surface_residues), n_idr))
        for n, rec in enumerate(self.surface_residues):
            surf_dev[n] = (
                self._weighted_row(
                    rec, idr_ctx, use_charge_weighting, use_aliphatic_weighting
                )
                - 2.0 * baseline
            )

        # how far the k-th residue from the window centre reaches (k = 0 .. half)
        reach = self._contact_reach(
            np.arange(half + 1), reach_b, reach_nu, contact_radius
        )

        # in cis, how far each IDR position can get from the anchor along the chain
        if tether is not None:
            tether_reach = self._contact_reach(
                self._contour_separation(n_idr, tether),
                reach_b,
                reach_nu,
                contact_radius,
            )
            anchor_pos = self.anchor_position(anchor)

        surf_row_of = {rec.key: n for n, rec in enumerate(self.surface_residues)}
        keys = [rec.key for rec in records]
        matrix = np.full((len(records), n_windows), np.nan)

        # where each IDR residue of the structure sits in its own IDR segment, and a
        # per-segment cache of weighted rows (only needed for rows='all')
        seg_pos = {}
        for si, seg in enumerate(self.idr_segments):
            for m, r in enumerate(seg["residues"]):
                seg_pos[r.key] = (si, m)
        seg_dev = {}

        for i, rec in enumerate(records):
            if rec.domain == "folded" and not rec.surface:
                continue
            if tether is not None and rec.position is None:
                continue

            if rec.surface:
                if use_footprint:
                    order, dists = self.footprint_order(rec.key)
                else:
                    # patch-only aggregation: the same set for every IDR position
                    patch = [
                        surf_row_of[k]
                        for k in self.surface_patch(rec.key)
                        if k in surf_row_of
                    ]
                    order = np.asarray(patch, dtype=int)
                    dists = np.zeros(len(order))
                dev = surf_dev[order]
            else:
                # An IDR residue of the structure has no surface footprint, but it is
                # not a point contact either: the challenge window lands on a stretch
                # of the structure's own disordered chain. The residue at contour
                # separation s from the row's residue sits a typical R(s) away, so it
                # is weighted on the same Gaussian falloff as the surface footprint
                # with R(s) standing in for the geodesic distance. This is what puts
                # IDR rows on the same scale as surface rows, and as an IDR:IDR
                # intermap cell, rather than leaving them as single-residue sums that
                # swamp the colour scale.
                si, m = seg_pos[rec.key]
                if si not in seg_dev:
                    seg_dev[si] = (
                        np.array(
                            [
                                self._weighted_row(
                                    r,
                                    idr_ctx,
                                    use_charge_weighting,
                                    use_aliphatic_weighting,
                                )
                                for r in self.idr_segments[si]["residues"]
                            ]
                        )
                        - 2.0 * baseline
                    )
                dev = seg_dev[si]
                sep = np.abs(np.arange(dev.shape[0]) - m)
                if use_footprint:
                    dists = self.reach_radius(sep, reach_b, reach_nu)
                else:
                    # patch-only: the flat mean over the window of the chain centred
                    # on the residue, which is exactly an IDR:IDR intermap cell
                    dev = dev[sep <= half]
                    dists = np.zeros(dev.shape[0])

            if dev.shape[0] == 0:
                continue

            # What the k-th IDR residue out from the window centre sees, for k = 0 .. half.
            # Each row of contact_w weights the surface (or the chain) by how likely that
            # IDR residue is to be in contact with it, and is normalised, so every row of
            # disc_mean is a weighted mean and the units of a cell are unchanged.
            if use_footprint:
                contact_w = np.exp(-1.5 * (dists[None, :] / reach[:, None]) ** 2)
                contact_w /= contact_w.sum(axis=1, keepdims=True)
                disc_mean = contact_w @ dev
            else:
                # patch-only: the same flat mean at every k
                disc_mean = np.repeat(dev.mean(axis=0)[None, :], half + 1, axis=0)

            # In cis, an IDR position is attenuated by how far it has to stray from the
            # anchor to get here. Same reasoning as the contact footprint above: the
            # tether's R(n) is a mean end-to-end distance, not a hard limit, so this
            # decays as a Gaussian rather than switching off at a cutoff, which would put
            # a sharp artificial boundary across the map of a large domain. Applied
            # before the windowed sum, so a partly-reachable window contributes
            # proportionately less.
            if tether is not None:
                dist = float(np.linalg.norm(anchor_pos - rec.position))
                reachability = np.exp(-1.5 * (dist / tether_reach) ** 2)
                disc_mean = disc_mean * reachability[None, :]

            # cell = sum over the window of the mean over what each IDR residue reaches
            total = disc_mean[0, half : n_idr - half].copy()
            for a in range(1, half + 1):
                total += disc_mean[a, half + a : n_idr - half + a]
                total += disc_mean[a, half - a : n_idr - half - a]

            matrix[i] = total

        return matrix, keys, window_centres

    def plot_surface_vs_idr_matrix(
        self,
        idr_sequence,
        window_size=31,
        rows="all",
        vmax=None,
        cmap="PRGn",
        buried_color="0.30",
        idr_offset=1,
        structure_offset=0,
        tic_frequency=10,
        xlabel="IDR residue",
        ylabel="structure residue",
        figsize=(7.0, 3.5),
        dpi=150,
        fname=None,
        ax=None,
        **kwargs,
    ):
        """
        Compute and draw a surface:IDR intermap.

        A thin wrapper that calls :meth:`surface_vs_idr_matrix` and hands the result to
        :func:`plot_surface_intermap`, so the map is drawn with a symmetric colour
        scale set from the data, buried residues greyed out and both axes labelled in
        residue numbers. Use the module-level function directly to draw a matrix you
        have already computed (for example a cis minus trans difference map).

        Parameters
        ----------
        idr_sequence : str
            The IDR amino acid sequence to challenge the surface with.

        window_size : int, optional
            Sliding-window size over the IDR, forwarded to
            :meth:`surface_vs_idr_matrix`. Default 31.

        rows : str, optional
            ``'all'`` (default) draws one row per residue in the structure, with buried
            residues greyed out; ``'surface'`` draws surface residues only. Note the
            default differs from :meth:`surface_vs_idr_matrix` (``'surface'``): a
            figure reads best straight down the chain, a matrix for analysis is usually
            wanted for the surface alone.

        vmax : float or None, optional
            Colour limit; the scale is symmetric, ``[-vmax, vmax]``. Default None sets
            it from the 99th percentile of the absolute finite values, so a handful of
            extreme cells cannot compress the rest of the map.

        cmap : str, optional
            Diverging matplotlib colormap. Default ``'PRGn'`` (purple attractive, green
            repulsive), matching the FINCHES IDR:IDR figures.

        buried_color : str, optional
            Colour for rows that cannot be scored (buried residues). Default ``'0.30'``.

        idr_offset : int, optional
            Added to the 0-based window-centre index to label the x axis. Default 1,
            so labels are 1-based residue numbers in ``idr_sequence``.

        structure_offset : int, optional
            Added to each row's ``resSeq`` to label the y axis, for a domain excised
            from a longer protein. Default 0.

        tic_frequency : int, optional
            Spacing, in residues, between axis tick labels. Default 10.

        xlabel, ylabel : str, optional
            Axis labels.

        figsize : tuple, optional
            Figure size in inches, used only when ``ax`` is None. Default (7, 3.5).

        dpi : int, optional
            Figure resolution, used only when ``ax`` is None. Default 150.

        fname : str or None, optional
            If given, the figure is saved to this path.

        ax : matplotlib.axes.Axes or None, optional
            Draw into an existing axes instead of creating a figure.

        **kwargs
            Any further keyword arguments (``anchor``, ``tether``, ``use_footprint``,
            ``reach_b``, ``reach_nu``, ``contact_radius``, ``use_charge_weighting``,
            ``use_aliphatic_weighting``) are forwarded to
            :meth:`surface_vs_idr_matrix`.

        Returns
        -------
        tuple
            ``(fig, ax, result)`` where ``result`` is the ``(matrix, keys,
            window_centres)`` tuple returned by :meth:`surface_vs_idr_matrix`.
        """
        result = self.surface_vs_idr_matrix(
            idr_sequence, window_size=window_size, rows=rows, **kwargs
        )
        fig, ax = plot_surface_intermap(
            *result,
            vmax=vmax,
            cmap=cmap,
            buried_color=buried_color,
            idr_offset=idr_offset,
            structure_offset=structure_offset,
            tic_frequency=tic_frequency,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            dpi=dpi,
            fname=fname,
            ax=ax,
        )
        return fig, ax, result

    def surface_vs_idr(
        self,
        idr_sequence,
        anchor=None,
        respect_reach=False,
        tether="N",
        reach_b=DEFAULT_REACH_B,
        reach_nu=DEFAULT_REACH_NU,
        use_charge_weighting=True,
        use_aliphatic_weighting=True,
    ):
        """
        Score every surface residue against an IDR sequence (capability 4).

        The interaction energetics use the FINCHES base pairwise lookup and the
        ``null_interaction_baseline`` from the frontend supplied at construction. The
        charge and aliphatic weighting replicate FINCHES, but for each surface residue
        the "local" environment is its contiguous-surface patch (capability 3) rather
        than the linear sequence window; for the IDR the standard +/-1 sequence window
        / sequence clusters are used. When ``respect_reach`` is enabled the polymer
        reach model (capability 5) restricts which IDR residues can contribute to each
        surface residue.

        Parameters
        ----------
        idr_sequence : str
            The IDR amino acid sequence to challenge the surface with.

        anchor : tuple, optional
            ``(chain_index, res_seq)`` of the structure residue at which the IDR is
            tethered. Required when ``respect_reach=True``.

        respect_reach : bool, optional
            If True, a surface residue only accumulates interaction from IDR residues
            that can reach it: those whose mean reach from the anchor,
            ``R(s) = reach_b * s**reach_nu`` for contour separation ``s``, is at least
            the straight-line distance from the anchor to the residue. This is a hard
            cutoff, which is what makes ``n_reachable`` a count; the cis mode of
            :meth:`surface_vs_idr_matrix` instead attenuates contributions smoothly.
            Default False.

        tether : str, optional
            Which terminus of ``idr_sequence`` is attached to ``anchor``, ``'N'`` or
            ``'C'``. Only consulted when ``respect_reach=True``. Default ``'N'``. See
            :meth:`idr_tether` to resolve this from an IDR segment.

        reach_b, reach_nu : float, optional
            Polymer-reach parameters ``R(s) = reach_b * s**reach_nu`` (Angstrom), where
            ``s`` is contour separation from the tether point.

        use_charge_weighting, use_aliphatic_weighting : bool, optional
            Toggle the FINCHES-style weighting terms. Default True.

        Returns
        -------
        dict
            Maps each surface residue key ``(chain_index, res_seq)`` to a dict with
            ``one_letter``, ``score`` and ``n_reachable`` (number of IDR residues that
            contributed). ``score`` is the FINCHES epsilon between the residue and the
            contributing IDR residues divided by their number, i.e. the mean over the
            IDR of ``value - 2 * null_interaction_baseline``, so it is on the same
            per-residue scale as an IDR:IDR epsilon: zero is poly(GS)-like, negative
            is attractive and positive is repulsive. Empty if the IDR sequence is empty
            or there are no surface residues.

        Raises
        ------
        ValueError
            If ``respect_reach=True`` but no ``anchor`` is provided.
        """
        surf = self.surface_residues
        if len(idr_sequence) == 0 or len(surf) == 0:
            return {}

        baseline = self.IMC_object.null_interaction_baseline
        idr_ctx = self._idr_context(idr_sequence)
        n_idr = len(idr_ctx["seq"])

        anchor_pos = None
        if respect_reach:
            if anchor is None:
                raise ValueError(
                    "respect_reach=True requires an anchor=(chain, resSeq)"
                )
            anchor_pos = self.anchor_position(anchor)
            # reach is indexed by contour separation from the tethered terminus
            reach = self.reach_radius(
                self._contour_separation(n_idr, tether), reach_b, reach_nu
            )

        results = {}
        for rec in surf:
            row = self._weighted_row(
                rec, idr_ctx, use_charge_weighting, use_aliphatic_weighting
            )

            if respect_reach:
                dist = float(np.linalg.norm(anchor_pos - rec.position))
                mask = reach >= dist
            else:
                mask = np.ones(n_idr, dtype=bool)

            n_reachable = int(mask.sum())
            if n_reachable == 0:
                results[rec.key] = {
                    "one_letter": rec.one_letter,
                    "score": 0.0,
                    "n_reachable": 0,
                }
                continue

            # FINCHES epsilon convention (see surface_vs_idr_matrix): every element
            # carries -2 * baseline, so a residue that behaves like poly(GS) towards
            # this IDR scores zero. The score is exactly the epsilon between this
            # residue (with its patch context) and the reachable IDR residues, per
            # IDR residue.
            score = float(np.mean(row[mask] - 2.0 * baseline))
            results[rec.key] = {
                "one_letter": rec.one_letter,
                "score": score,
                "n_reachable": n_reachable,
            }
        return results

    def _weighted_row(
        self,
        rec,
        idr_ctx,
        use_charge_weighting=True,
        use_aliphatic_weighting=True,
    ):
        """
        FINCHES-weighted interaction of one structure residue with every IDR residue.

        This is the single-row analogue of
        ``InteractionMatrixConstructor.calculate_weighted_pairwise_matrix``. The base
        pairwise energies come from the IMC lookup; the charge weighting reduces
        like-charge repulsion between two charged residues by the charge weight
        (``charge_weight_from_counts``) of their combined local contexts, scaled by
        the IMC's charge prefactor; and the aliphatic weighting multiplies each pair
        by ``aliphatic_weight_from_levels`` of the two clustering levels. The IDR's
        context comes from its sequence; the structure residue's comes from
        :meth:`_residue_context` (its surface patch if it is a surface residue, its
        linear sequence window otherwise).

        Parameters
        ----------
        rec : Residue
            The structure residue.

        idr_ctx : dict
            The IDR context returned by :meth:`_idr_context`.

        use_charge_weighting, use_aliphatic_weighting : bool, optional
            Toggle each weighting term. Default True.

        Returns
        -------
        numpy.ndarray
            Weighted interaction value of ``rec`` against each IDR position, length
            ``len(idr_ctx["seq"])``. The null baseline has not been subtracted.
        """
        imc = self.IMC_object
        aa = rec.one_letter
        lookup_row = imc.lookup[aa]

        # the residue interacts as itself; neighbouring chemistry enters through the
        # contact footprint in surface_vs_idr_matrix() and through the local-context
        # weighting below
        w = np.array([lookup_row[r] for r in idr_ctx["seq"]], dtype=float)

        s_pos, s_neg, s_level = self._residue_context(rec)

        if (
            use_charge_weighting
            and imc.charge_prefactor is not None
            and aa in CHARGED_RESIDUES
        ):
            # only IDR positions that are themselves charged get the weighting
            weight = charge_weight_from_counts(
                s_pos + idr_ctx["pos"], s_neg + idr_ctx["neg"]
            )
            w = w - w * (weight * idr_ctx["charged"]) * imc.charge_prefactor

        if use_aliphatic_weighting:
            w = w * aliphatic_weight_from_levels(s_level, idr_ctx["ali"])

        return w

    def surface_vs_surface(
        self,
        use_charge_weighting=True,
        use_aliphatic_weighting=True,
    ):
        """
        Score interactions between all pairs of surface residues (capability 4).

        Each surface residue carries its own surface-patch context, so the
        charge/aliphatic weighting reflects both residues' local surface environments.
        The base energetics and weighting forms match FINCHES (and
        :meth:`surface_vs_idr`), using the frontend supplied at construction.

        Parameters
        ----------
        use_charge_weighting, use_aliphatic_weighting : bool, optional
            Toggle the FINCHES-style weighting terms. Default True.

        Returns
        -------
        dict
            Maps each surface residue pair ``(key_a, key_b)`` (with ``key_a`` earlier
            in surface order than ``key_b``) to the interaction value on the FINCHES
            epsilon scale (``value - 2 * null_interaction_baseline``, so zero is
            poly(GS)-like and negative is attractive).
        """
        surf = self.surface_residues
        imc = self.IMC_object
        baseline = imc.null_interaction_baseline
        prefactor = imc.charge_prefactor

        # patch context (n_pos, n_neg, aliphatic level) per surface residue
        ctx = {rec.key: self._residue_context(rec) for rec in surf}

        results = {}
        for i in range(len(surf)):
            ra = surf[i]
            pa, na, la = ctx[ra.key]
            for j in range(i + 1, len(surf)):
                rb = surf[j]
                pb, nb, lb = ctx[rb.key]
                val = float(imc.lookup[ra.one_letter][rb.one_letter])

                if (
                    use_charge_weighting
                    and prefactor is not None
                    and ra.one_letter in CHARGED_RESIDUES
                    and rb.one_letter in CHARGED_RESIDUES
                ):
                    val = (
                        val
                        - val * charge_weight_from_counts(pa + pb, na + nb) * prefactor
                    )

                if use_aliphatic_weighting:
                    val = val * aliphatic_weight_from_levels(la, lb)

                # FINCHES epsilon convention: -2 * baseline, so zero is poly(GS)-like
                results[(ra.key, rb.key)] = val - 2.0 * baseline
        return results

    # ------------------------------------------------------------------
    # polymer reach model (capability 5)
    # ------------------------------------------------------------------
    def reach_radius(self, n, reach_b=DEFAULT_REACH_B, reach_nu=DEFAULT_REACH_NU):
        """
        Maximum reach (Angstrom) of the n-th IDR residue from its anchor.

        Uses the simple polymer scaling ``R(n) = reach_b * n**reach_nu``, i.e. the
        approximate end-to-end distance of an ``n``-residue tether.

        Parameters
        ----------
        n : int or array-like
            Residue index (1-based) along the IDR, counted outward from the anchor.
        reach_b, reach_nu : float, optional
            Polymer-reach parameters; ``R(n) = reach_b * n**reach_nu``.

        Returns
        -------
        float or numpy.ndarray
            Reach radius in Angstrom (array-like if ``n`` is array-like).
        """
        return reach_b * np.power(np.asarray(n, dtype=float), reach_nu)

    def _contact_reach(
        self,
        n,
        reach_b=DEFAULT_REACH_B,
        reach_nu=DEFAULT_REACH_NU,
        contact_radius=DEFAULT_CONTACT_RADIUS,
    ):
        """
        Polymer reach floored by the contact shell of a single residue.

        ``R(n)`` on its own treats a residue as a point, so ``R(0)`` is zero and a
        zero-length tether reaches nothing. In reality a residue has a side chain and
        touches a shell of radius ``contact_radius`` wherever it sits. The two are
        independent displacements, so they add in quadrature.

        Parameters
        ----------
        n : int or array-like
            Number of residues of chain between the two points.
        reach_b, reach_nu : float, optional
            Polymer-reach parameters; see :meth:`reach_radius`.
        contact_radius : float, optional
            Radius of a single residue's contact shell, in Angstrom.

        Returns
        -------
        numpy.ndarray
            Effective reach in Angstrom, never exactly zero so it is safe to divide by.
        """
        return np.clip(
            np.hypot(self.reach_radius(n, reach_b, reach_nu), contact_radius),
            1e-9,
            None,
        )

    def reachable_surface_residues(
        self, anchor, idr_length, reach_b=DEFAULT_REACH_B, reach_nu=DEFAULT_REACH_NU
    ):
        """
        Surface residues reachable by an IDR anchored at a structure residue.

        A surface residue is reachable if its straight-line distance from the anchor is
        no greater than the mean reach of the IDR's final residue, ``R(idr_length)``.
        This is the same hard cutoff as ``surface_vs_idr(respect_reach=True)``; for
        each reachable residue the smallest IDR index able to reach it is also
        returned.

        Parameters
        ----------
        anchor : tuple
            ``(chain_index, res_seq)`` of the anchoring residue.
        idr_length : int
            Number of residues in the IDR.
        reach_b, reach_nu : float, optional
            Polymer-reach parameters; ``R(n) = reach_b * n**reach_nu``.

        Returns
        -------
        dict
            Maps each reachable surface residue key to the minimum IDR residue index
            (1-based) able to reach it.
        """
        anchor_pos = self.anchor_position(anchor)
        max_reach = self.reach_radius(idr_length, reach_b, reach_nu)

        reachable = {}
        for rec in self.surface_residues:
            dist = float(np.linalg.norm(anchor_pos - rec.position))
            if dist <= max_reach:
                # smallest i with reach_b*i**nu >= dist
                i_min = (
                    int(np.ceil((dist / reach_b) ** (1.0 / reach_nu)))
                    if dist > 0
                    else 1
                )
                i_min = max(1, min(i_min, idr_length))
                reachable[rec.key] = i_min
        return reachable

    # ------------------------------------------------------------------
    # PDB output (beta / B-factor column carries an annotation)
    # ------------------------------------------------------------------
    def _resolve_key(self, pos):
        """
        Resolve a residue identifier to a ``(chain_index, res_seq)`` key.

        Parameters
        ----------
        pos : tuple or int
            Either a ``(chain_index, res_seq)`` key, or a bare ``res_seq`` integer
            (only allowed when the structure has a single protein chain).

        Returns
        -------
        tuple of (int, int)
            The resolved residue key.

        Raises
        ------
        TypeError
            If ``pos`` is neither an int nor a 2-tuple.
        ValueError
            If a bare integer is given for a multi-chain structure.
        KeyError
            If no residue with the resolved key exists.
        """
        if isinstance(pos, tuple):
            key = (int(pos[0]), int(pos[1]))
        elif isinstance(pos, (int, np.integer)):
            chains = sorted({r.chain_index for r in self.residues})
            if len(chains) != 1:
                raise ValueError(
                    "pos was given as an integer but the structure has multiple "
                    "chains; pass a (chain_index, res_seq) tuple instead."
                )
            key = (chains[0], int(pos))
        else:
            raise TypeError(
                "pos must be an int resSeq or a (chain_index, res_seq) tuple"
            )
        if key not in self._by_key:
            raise KeyError(f"no residue {key} in structure")
        return key

    def _write_pdb_with_bfactors(self, filename, residue_values, default=0.0):
        """
        Write the structure to a PDB file with per-residue values in the B-factor
        (beta) column.

        Every atom of a residue receives that residue's value; residues absent from
        ``residue_values`` (and any non-protein atoms) receive ``default``. Only
        modeled residues are written (residues missing from the coordinates have no
        atoms to output).

        Parameters
        ----------
        filename : str
            Output PDB path.
        residue_values : dict
            Maps residue key ``(chain_index, res_seq)`` to the value to place in the
            beta column.
        default : float, optional
            Value for atoms whose residue is not in ``residue_values``. Default 0.0.

        Returns
        -------
        str
            ``filename``.
        """
        idx_to_rec = {r.md_index: r for r in self.residues if r.modeled}
        bfactors = np.full(self.traj.n_atoms, float(default), dtype=float)
        for atom in self.traj.topology.atoms:
            rec = idx_to_rec.get(atom.residue.index)
            if rec is not None and rec.key in residue_values:
                bfactors[atom.index] = float(residue_values[rec.key])
        self.traj.save_pdb(filename, force_overwrite=True, bfactors=bfactors)
        return filename

    def write_pdb_solvent_accessibility(self, filename="solvent_accessibility.pdb"):
        """
        Write a PDB whose beta column flags solvent-accessible residues.

        The beta column is 1.0 for solvent-accessible residues and 0.0 otherwise.
        All modeled residues are included: folded-domain surface residues and every
        IDR residue (IDRs are intrinsically solvent-exposed) are marked accessible,
        while buried folded-domain residues are marked 0.0.

        Parameters
        ----------
        filename : str, optional
            Output PDB path. Default ``'solvent_accessibility.pdb'``.

        Returns
        -------
        str
            The path written.
        """
        values = {}
        for rec in self.residues:
            if not rec.modeled:
                continue
            accessible = rec.surface or rec.domain == "idr"
            values[rec.key] = 1.0 if accessible else 0.0
        return self._write_pdb_with_bfactors(filename, values, default=0.0)

    def write_pdb_groups(self, pos, filename="groups.pdb", include_self=True):
        """
        Write a PDB whose beta column flags the surface neighbours of a residue.

        The beta column is 1.0 for the contiguous-surface neighbours of ``pos`` (the
        residues linked to it in the surface net) and 0.0 for all other residues.

        Parameters
        ----------
        pos : tuple or int
            The central surface residue, as a ``(chain_index, res_seq)`` key (or a
            bare ``res_seq`` for a single-chain structure).
        filename : str, optional
            Output PDB path. Default ``'groups.pdb'``.
        include_self : bool, optional
            If True (default), ``pos`` itself is also set to 1.0 so the whole local
            group (centre + neighbours) is highlighted; if False only the neighbours
            are flagged.

        Returns
        -------
        str
            The path written.

        Raises
        ------
        ValueError
            If ``pos`` is not a folded-domain surface residue (the surface net, and
            hence neighbours, is only defined for surface residues).
        """
        key = self._resolve_key(pos)
        if not self._by_key[key].surface:
            raise ValueError(
                f"residue {key} is not a folded-domain surface residue; the surface "
                "net (and its neighbours) is only defined for surface residues."
            )
        group = set(self.surface_neighbours.get(key, []))
        if include_self:
            group.add(key)
        values = {k: 1.0 for k in group}
        return self._write_pdb_with_bfactors(filename, values, default=0.0)

    def write_pdb_idr_interaction(
        self,
        seq,
        filename="idr_interaction.pdb",
        use_charge_weighting=True,
        use_aliphatic_weighting=True,
    ):
        """
        Write a PDB whose beta column reports per-residue interaction with an IDR.

        For each folded-domain surface residue the beta column is set to its FINCHES
        interaction score against the peptide/IDR ``seq`` (negative = attractive; see
        :meth:`surface_vs_idr`), using the frontend supplied at construction. All
        other residues (buried, IDR, missing) are set to 0.0.

        Parameters
        ----------
        seq : str
            The IDR / peptide sequence to score the surface against.
        filename : str, optional
            Output PDB path. Default ``'idr_interaction.pdb'``.
        use_charge_weighting, use_aliphatic_weighting : bool, optional
            Toggle the FINCHES-style weighting terms (forwarded to
            :meth:`surface_vs_idr`). Default True.

        Returns
        -------
        str
            The path written.

        Notes
        -----
        Visualisation tools read the beta column as a float, so the attractive
        (negative) and repulsive (positive) scores can be colour-mapped directly.
        """
        scores = self.surface_vs_idr(
            seq,
            use_charge_weighting=use_charge_weighting,
            use_aliphatic_weighting=use_aliphatic_weighting,
        )
        values = {key: info["score"] for key, info in scores.items()}
        return self._write_pdb_with_bfactors(filename, values, default=0.0)

    # the original capitalisation, kept so existing code keeps working
    write_pdb_IDR_interaction = write_pdb_idr_interaction

    def write_pdb_accessible_from_position(
        self,
        pos,
        filename="accessible_from_position.pdb",
        reach_b=DEFAULT_REACH_B,
        reach_nu=DEFAULT_REACH_NU,
    ):
        """
        Write a PDB flagging surface residues reachable from an IDR position.

        ``pos`` must lie in an IDR. The polymer-reach model is used to determine which
        folded-domain surface residues that IDR position can physically contact: the
        reach radius is ``R(k) = reach_b * k**reach_nu`` where ``k`` is the number of
        residues from ``pos`` to its folded-domain anchor. If the IDR is flanked by
        two folded anchors (an internal loop), reachability from either anchor is
        taken (the union). The beta column is 1.0 for reachable surface residues and
        0.0 otherwise.

        Parameters
        ----------
        pos : tuple or int
            The IDR residue, as a ``(chain_index, res_seq)`` key (or a bare
            ``res_seq`` for a single-chain structure).
        filename : str, optional
            Output PDB path. Default ``'accessible_from_position.pdb'``.
        reach_b, reach_nu : float, optional
            Polymer-reach parameters; ``R(k) = reach_b * k**reach_nu``.

        Returns
        -------
        str
            The path written.

        Raises
        ------
        ValueError
            If ``pos`` is in a folded domain, or if its IDR segment has no
            folded-domain anchor from which reach can be measured.
        """
        key = self._resolve_key(pos)
        rec = self._by_key[key]
        if rec.domain == "folded":
            raise ValueError(
                f"residue {key} is in a folded domain; accessibility-from-position "
                "is only defined for IDR residues."
            )

        # locate the IDR segment containing pos and pos's index within it
        seg = None
        idx_in_seg = None
        for s in self.idr_segments:
            for i, r in enumerate(s["residues"]):
                if r.key == key:
                    seg, idx_in_seg = s, i
                    break
            if seg is not None:
                break
        if seg is None:
            raise ValueError(f"could not locate IDR segment for residue {key}")

        if seg["n_anchor"] is None and seg["c_anchor"] is None:
            raise ValueError(
                f"the IDR segment containing residue {key} has no folded-domain "
                "anchor; reach cannot be measured."
            )

        n_res = len(seg["residues"])
        reachable = {}
        # distance (in residues) from each flanking anchor outward to pos
        if seg["n_anchor"] is not None:
            k_n = idx_in_seg + 1
            for skey, imin in self.reachable_surface_residues(
                seg["n_anchor"].key, k_n, reach_b, reach_nu
            ).items():
                reachable[skey] = min(reachable.get(skey, imin), imin)
        if seg["c_anchor"] is not None:
            k_c = n_res - idx_in_seg
            for skey, imin in self.reachable_surface_residues(
                seg["c_anchor"].key, k_c, reach_b, reach_nu
            ).items():
                reachable[skey] = min(reachable.get(skey, imin), imin)

        values = {k: 1.0 for k in reachable}
        return self._write_pdb_with_bfactors(filename, values, default=0.0)


def plot_surface_intermap(
    matrix,
    keys,
    window_centres,
    vmax=None,
    cmap="PRGn",
    buried_color="0.30",
    idr_offset=1,
    structure_offset=0,
    tic_frequency=10,
    xlabel="IDR residue",
    ylabel="structure residue",
    figsize=(7.0, 3.5),
    dpi=150,
    fname=None,
    ax=None,
):
    """
    Draw a surface:IDR intermap returned by :meth:`StructureSurface.surface_vs_idr_matrix`.

    Rows are structure residues (in the order of ``keys``) and columns are IDR
    windows. The colour scale is symmetric about zero so attractive (negative) and
    repulsive (positive) cells are directly comparable, and by default its limit is
    set from the data rather than from the most extreme cell, which is what keeps the
    dynamic range readable when a few cells are far larger than the rest. Rows that
    are all ``np.nan`` (buried residues under ``rows='all'``) are drawn in
    ``buried_color`` so they read as "no exposed surface" rather than as zero. When
    the rows span more than one chain a thin black line separates the chains and the
    y tick labels carry the chain index.

    Parameters
    ----------
    matrix : numpy.ndarray
        The map, shape ``(n_rows, n_windows)``.

    keys : list of tuple
        ``(chain_index, res_seq)`` for each row, as returned with the matrix.

    window_centres : numpy.ndarray
        0-based index into the IDR of each window's centre residue, as returned with
        the matrix.

    vmax : float or None, optional
        Colour limit; the scale is ``[-vmax, vmax]``. Default None uses the 99th
        percentile of the absolute finite values.

    cmap : str, optional
        Diverging matplotlib colormap. Default ``'PRGn'``.

    buried_color : str, optional
        Colour for all-``nan`` rows. Default ``'0.30'``.

    idr_offset : int, optional
        Added to ``window_centres`` for the x tick labels. Default 1 (1-based residue
        numbers).

    structure_offset : int, optional
        Added to each row's ``res_seq`` for the y tick labels. Default 0.

    tic_frequency : int, optional
        Spacing in residues between tick labels. Default 10.

    xlabel, ylabel : str, optional
        Axis labels.

    figsize : tuple, optional
        Figure size in inches when a new figure is created. Default (7, 3.5).

    dpi : int, optional
        Figure resolution when a new figure is created. Default 150.

    fname : str or None, optional
        If given, save the figure to this path.

    ax : matplotlib.axes.Axes or None, optional
        Draw into this axes instead of creating a new figure.

    Returns
    -------
    tuple
        ``(fig, ax)``.

    Raises
    ------
    ValueError
        If the matrix is empty or has no finite values.
    """
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        raise ValueError("cannot plot an empty intermap")
    finite = np.isfinite(matrix)
    if not finite.any():
        raise ValueError("the intermap has no finite values to plot")

    if vmax is None:
        vmax = float(np.percentile(np.abs(matrix[finite]), 99))
        if vmax <= 0:
            vmax = float(np.max(np.abs(matrix[finite]))) or 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(buried_color)

    n_rows = matrix.shape[0]
    x0 = int(window_centres[0]) + idr_offset
    x1 = int(window_centres[-1]) + idr_offset + 1

    im = ax.imshow(
        np.ma.masked_invalid(matrix),
        extent=[x0, x1, 0, n_rows],
        origin="lower",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        cmap=cm,
        interpolation="nearest",
    )

    # x axis: IDR residue numbers
    xt = np.arange(x0, x1, tic_frequency)
    ax.set_xticks(xt)
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel(xlabel)

    # y axis: structure residue numbers, one label every tic_frequency rows; chain
    # index is shown only when there is more than one chain
    chains = sorted({k[0] for k in keys})
    row_positions = np.arange(0, n_rows, tic_frequency)
    if len(chains) > 1:
        labels = [
            f"{keys[i][0]}:{keys[i][1] + structure_offset}" for i in row_positions
        ]
    else:
        labels = [str(keys[i][1] + structure_offset) for i in row_positions]
    ax.set_yticks(row_positions + 0.5)
    ax.set_yticklabels(labels)
    ax.set_ylabel(ylabel)

    # separate chains
    for i in range(1, n_rows):
        if keys[i][0] != keys[i - 1][0]:
            ax.axhline(i, color="k", linewidth=0.8)

    ax.set_xlim(x0, x1)
    ax.set_ylim(0, n_rows)
    fig.colorbar(im, ax=ax, pad=0.02)

    if fname is not None:
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")

    return fig, ax
