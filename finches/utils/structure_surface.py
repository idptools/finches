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

The existing ``finches.utils.folded_domain_utils.FoldedDomain`` class is intentionally
left untouched; this module is additive.

By: Alex Holehouse & contributors
"""

import numpy as np
import mdtraj as md
import networkx as nx
from scipy.spatial import cKDTree
import metapredict as meta

from finches.utils.folded_domain_utils import MAX_SASA_DATA, THREE_TO_ONE


# residue chemistry groups (match finches.parsing_aminoacid_sequences)
POSITIVE = {"R", "K"}
NEGATIVE = {"E", "D"}
CHARGED = POSITIVE | NEGATIVE
ALIPHATIC = {"A", "V", "I", "L", "M"}

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


def _load_structure(path):
    """
    Load a protein structure from a PDB or mmCIF/PDBx file with mdtraj.

    PDB files are read with ``mdtraj.load_pdb`` and all other extensions (including
    ``.cif``/``.mmcif``/``.pdbx``) are dispatched through ``mdtraj.load``, which
    selects the appropriate reader. If the chosen reader fails, ``mdtraj.load`` is
    tried as a last resort.

    Parameters
    ----------
    path : str
        Path to the structure file.

    Returns
    -------
    mdtraj.Trajectory
        The loaded structure (single frame for a static model).
    """
    lower = str(path).lower()
    try:
        if lower.endswith(_PDB_EXTENSIONS):
            return md.load_pdb(path)
        return md.load(path)
    except Exception:
        # last resort: let mdtraj infer the format itself
        return md.load(path)


class Residue:
    """
    Lightweight per-residue record used internally by :class:`StructureSurface`.

    Attributes
    ----------
    chain_index : int
        0-based mdtraj chain index.
    res_seq : int
        Author residue number (PDB ``resSeq``). For residues that are missing from
        the coordinates a synthetic 1-based number derived from the sequence position
        is used.
    seq_index : int
        0-based position of this residue in the full chain sequence.
    one_letter : str
        One-letter amino acid code.
    modeled : bool
        True if the residue has coordinates in the structure.
    domain : str
        Either ``'folded'`` or ``'idr'``.
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
        "md_index",
        "position",
        "surface",
    )

    def __init__(self, chain_index, res_seq, seq_index, one_letter, modeled):
        """
        Create a residue record.

        Newly created records default to ``domain='idr'``, ``surface=False``,
        ``md_index=None`` and ``position=None``; these are populated by
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
        numbering gaps filled by placeholder glycines and recorded as missing
        residues in ``self._gap_positions``.

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

        Raises
        ------
        ValueError
            If a provided sequence is too short for a modeled ``resSeq`` or a residue
            identity does not match the provided sequence.
        """
        modeled = [
            (r.resSeq, THREE_TO_ONE.get(r.name, "G"), r.index)
            for r in chain.residues
            if r.is_protein
        ]

        provided = self._provided_sequences.get(chain.index)
        if provided is not None:
            for res_seq, aa, _ in modeled:
                pos = res_seq - 1
                if pos < 0 or pos >= len(provided):
                    raise ValueError(
                        f"Chain {chain.index}: residue resSeq={res_seq} falls "
                        f"outside the provided sequence (length {len(provided)}). "
                        "resSeq is assumed to be 1-based into the sequence."
                    )
                if aa != "G" and provided[pos] != aa:
                    raise ValueError(
                        f"Chain {chain.index}: residue identity mismatch at "
                        f"resSeq={res_seq}: structure has '{aa}' but the provided "
                        f"sequence has '{provided[pos]}'."
                    )
            return provided, modeled, lambda rs: rs - 1

        if not modeled:
            return "", modeled, lambda rs: rs

        # reconstruct from modeled residues, filling resSeq gaps as missing (IDR)
        min_rs = modeled[0][0]
        max_rs = modeled[-1][0]
        seqlist = ["G"] * (max_rs - min_rs + 1)
        present = set()
        for res_seq, aa, _ in modeled:
            seqlist[res_seq - min_rs] = aa
            present.add(res_seq)
        full = "".join(seqlist)
        self._gap_positions = getattr(self, "_gap_positions", {})
        self._gap_positions[chain.index] = {
            rs for rs in range(min_rs, max_rs + 1) if rs not in present
        }
        return full, modeled, lambda rs: rs - min_rs

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

            full, modeled, map_resseq = self._chain_full_sequence(chain)
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
                    # missing residue: use a synthetic resSeq from the sequence index
                    res_seq = seq_index + 1
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
        a chain, its modeled region is treated as entirely folded. The contiguous IDR
        runs and their anchors are then recorded on ``self.idr_segments``.

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
            except Exception:
                # if metapredict fails, treat the whole modeled region as folded
                folded_positions = set(range(len(full)))

            for rec in self._chain_residues(chain_index):
                # a residue is part of a folded domain only if it is BOTH modeled
                # and predicted folded; missing residues are always IDR
                if rec.modeled and rec.seq_index in folded_positions:
                    rec.domain = "folded"
                else:
                    rec.domain = "idr"

        # record IDR segments (contiguous IDR runs) with their folded anchors
        self.idr_segments = self._find_idr_segments()

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
        n_pos = sum(1 for k in patch if self._by_key[k].one_letter in POSITIVE)
        n_neg = sum(1 for k in patch if self._by_key[k].one_letter in NEGATIVE)
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
        if center.one_letter not in ALIPHATIC:
            return 0
        # count of aliphatic residues in the patch (including self), capped at 3
        count = sum(
            1
            for k in self.surface_patch(key)
            if self._by_key[k].one_letter in ALIPHATIC
        )
        return min(count, 3)

    def _ali_weight(self, level_a, level_b):
        """
        FINCHES aliphatic pairwise weight from two clustering levels.

        Reproduces the tiered weighting of
        ``finches.parsing_aminoacid_sequences.get_aliphatic_weighted_mask``.

        Parameters
        ----------
        level_a, level_b : int
            Aliphatic clustering levels of the two partners.

        Returns
        -------
        float
            ``3.0`` if ``min(level_a, level_b) >= 3``, ``1.5`` if it equals 2,
            otherwise ``1.0``.
        """
        m = min(level_a, level_b)
        if m >= 3:
            return 3.0
        if m == 2:
            return 1.5
        return 1.0

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
            pos_w, neg_w = self._seq_charge_windows(seq)
            self._linear_context_cache[chain_index] = (
                pos_w,
                neg_w,
                self._seq_aliphatic_groups(seq),
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

    def _seq_charge_windows(self, sequence):
        """
        Per-position charged-residue counts over the +/-1 window of a sequence.

        This is the linear-sequence analogue of :meth:`_patch_charge_counts`, used for
        the IDR partner. It is computed as a width-3 moving sum over charge-indicator
        arrays (matching ``finches.parsing_aminoacid_sequences.get_charge_weighted_mask``).

        Parameters
        ----------
        sequence : str
            The (converted) amino acid sequence.

        Returns
        -------
        tuple of numpy.ndarray
            ``(pos_window, neg_window)``, each of length ``len(sequence)``, giving the
            number of positive / negative residues in the +/-1 window of each position.
        """
        pos = np.array([1.0 if r in POSITIVE else 0.0 for r in sequence])
        neg = np.array([1.0 if r in NEGATIVE else 0.0 for r in sequence])
        n = len(sequence)
        if n == 0:
            return pos, neg
        kernel = np.ones(3)
        pos_w = np.convolve(pos, kernel, mode="full")[1 : n + 1]
        neg_w = np.convolve(neg, kernel, mode="full")[1 : n + 1]
        return pos_w, neg_w

    def _seq_aliphatic_groups(self, sequence):
        """
        Per-position aliphatic clustering level for a linear sequence.

        Delegates to ``finches.parsing_aminoacid_sequences.get_aliphatic_groups`` so
        the IDR partner uses exactly the same clustering definition as FINCHES.

        Parameters
        ----------
        sequence : str
            The (converted) amino acid sequence.

        Returns
        -------
        numpy.ndarray
            Per-position clustering levels (0/1/2/3), length ``len(sequence)``.
        """
        from finches import parsing_aminoacid_sequences

        return np.array(parsing_aminoacid_sequences.get_aliphatic_groups(sequence))

    def _anchor_position(self, anchor):
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

    def _footprint_order(self, key):
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
            greyed out. IDR residues are scored against themselves only (they have no
            surface footprint) using linear-sequence weighting context.

        use_footprint : bool, optional
            If True (default), aggregate over the window-sized contact footprint
            described above. If False, aggregate over the contiguous-surface patch alone
            for every IDR residue, which reduces the cell exactly to
            ``IMC.calculate_epsilon_value(window_sequence, patch_sequence)``.

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

        imc = self.IMC_object
        baseline = imc.null_interaction_baseline

        idr = imc.sequence_converter(idr_sequence)
        idr_pos, idr_neg = self._seq_charge_windows(idr)
        idr_ali = self._seq_aliphatic_groups(idr)

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
        # has to match for these values to be comparable with an IDR:IDR intermap.
        surf_dev = np.zeros((len(self.surface_residues), n_idr))
        for n, rec in enumerate(self.surface_residues):
            row = self._weighted_row(
                rec,
                idr,
                idr_pos,
                idr_neg,
                idr_ali,
                imc.lookup,
                imc.charge_prefactor,
                use_charge_weighting,
                use_aliphatic_weighting,
            )
            surf_dev[n] = np.where(
                row == baseline, -2.0 * baseline, row - 2.0 * baseline
            )

        if contact_radius < 0:
            raise ValueError(f"contact_radius must be >= 0, got {contact_radius}")

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
            anchor_pos = self._anchor_position(anchor)

        surf_row_of = {rec.key: n for n, rec in enumerate(self.surface_residues)}
        keys = [rec.key for rec in records]
        matrix = np.full((len(records), n_windows), np.nan)

        for i, rec in enumerate(records):
            if rec.domain == "folded" and not rec.surface:
                continue
            if tether is not None and rec.position is None:
                continue

            if rec.surface:
                if use_footprint:
                    order, dists = self._footprint_order(rec.key)
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
                # an IDR residue has no surface footprint; it interacts as itself
                dev = self._weighted_row(
                    rec,
                    idr,
                    idr_pos,
                    idr_neg,
                    idr_ali,
                    imc.lookup,
                    imc.charge_prefactor,
                    use_charge_weighting,
                    use_aliphatic_weighting,
                )
                dev = np.where(dev == baseline, -2.0 * baseline, dev - 2.0 * baseline)[
                    None, :
                ]
                dists = np.zeros(1)

            if dev.shape[0] == 0:
                continue

            # What the k-th IDR residue out from the window centre sees, for k = 0 .. half.
            # Each row of contact_w weights the surface by how likely that IDR residue is
            # to be in contact with it, and is normalised, so every row of disc_mean is a
            # weighted mean over the surface and the units of a cell are unchanged.
            if use_footprint and rec.surface:
                contact_w = np.exp(-1.5 * (dists[None, :] / reach[:, None]) ** 2)
                contact_w /= contact_w.sum(axis=1, keepdims=True)
                disc_mean = contact_w @ dev
            else:
                # patch-only (or a lone IDR residue): the same flat mean at every k
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
            that can physically reach it given the polymer-reach model. Default False.

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
            ``one_letter``, ``score`` (mean-field interaction value; negative =
            attractive), and ``n_reachable`` (number of IDR residues that contributed).
            Empty if the IDR sequence is empty or there are no surface residues.

        Raises
        ------
        ValueError
            If ``respect_reach=True`` but no ``anchor`` is provided.
        """
        surf = self.surface_residues
        if len(idr_sequence) == 0 or len(surf) == 0:
            return {}

        imc = self.IMC_object
        baseline = imc.null_interaction_baseline
        prefactor = imc.charge_prefactor
        lookup = imc.lookup

        idr = imc.sequence_converter(idr_sequence)
        idr_pos, idr_neg = self._seq_charge_windows(idr)
        idr_ali = self._seq_aliphatic_groups(idr)

        anchor_pos = None
        if respect_reach:
            if anchor is None:
                raise ValueError(
                    "respect_reach=True requires an anchor=(chain, resSeq)"
                )
            anchor_pos = self._anchor_position(anchor)
            # reach is indexed by contour separation from the tethered terminus
            reach = self.reach_radius(
                self._contour_separation(len(idr), tether), reach_b, reach_nu
            )

        results = {}
        for rec in surf:
            row = self._weighted_row(
                rec,
                idr,
                idr_pos,
                idr_neg,
                idr_ali,
                lookup,
                prefactor,
                use_charge_weighting,
                use_aliphatic_weighting,
            )

            if respect_reach:
                dist = float(np.linalg.norm(anchor_pos - rec.position))
                mask = reach >= dist
            else:
                mask = np.ones(len(idr), dtype=bool)

            n_reachable = int(mask.sum())
            if n_reachable == 0:
                results[rec.key] = {
                    "one_letter": rec.one_letter,
                    "score": 0.0,
                    "n_reachable": 0,
                }
                continue

            sub = row[mask]
            attractive = sub[sub < baseline] - baseline
            repulsive = sub[sub > baseline] - baseline
            score = float((attractive.sum() + repulsive.sum()) / n_reachable)
            results[rec.key] = {
                "one_letter": rec.one_letter,
                "score": score,
                "n_reachable": n_reachable,
            }
        return results

    def _weighted_row(
        self,
        surf_rec,
        idr,
        idr_pos,
        idr_neg,
        idr_ali,
        lookup,
        prefactor,
        use_charge_weighting,
        use_aliphatic_weighting,
    ):
        """
        Compute the FINCHES-weighted interaction values for one surface residue.

        This is the single-row analogue of FINCHES'
        ``calculate_weighted_pairwise_matrix``: the base pairwise energies come from
        the IMC lookup, charge weighting reduces like-charge repulsion using the
        combined surface-patch and IDR-window charge composition, and aliphatic
        weighting up-weights aliphatic-cluster contacts.

        Parameters
        ----------
        surf_rec : Residue
            The surface residue (provides the patch context).
        idr : str
            The (converted) IDR sequence.
        idr_pos, idr_neg : numpy.ndarray
            Per-position +/-1-window positive/negative counts for the IDR (from
            :meth:`_seq_charge_windows`).
        idr_ali : numpy.ndarray
            Per-position aliphatic clustering levels for the IDR.
        lookup : dict
            The IMC nested base pairwise interaction lookup.
        prefactor : float or None
            The IMC charge prefactor (charge weighting is skipped if None).
        use_charge_weighting, use_aliphatic_weighting : bool
            Toggle each weighting term.

        Returns
        -------
        numpy.ndarray
            Weighted interaction value of ``surf_rec`` against each IDR position,
            length ``len(idr)``.
        """
        surf_aa = surf_rec.one_letter

        # the residue interacts as itself; neighbouring chemistry enters through the
        # contact footprint in surface_vs_idr_matrix(), and through the patch-derived
        # weighting context below
        base = np.array([lookup[surf_aa][idr[j]] for j in range(len(idr))], dtype=float)
        w = base.copy()

        # surface residues take their context from their 3D surface patch; IDR residues
        # have no patch and fall back to the linear sequence window
        s_pos, s_neg, s_level = self._residue_context(surf_rec)

        if use_charge_weighting and prefactor is not None and surf_aa in CHARGED:
            total_pos = s_pos + idr_pos
            total_neg = s_neg + idr_neg
            total_charge = total_pos + total_neg
            denom = np.where(total_charge == 0, 1.0, total_charge)
            charge_weight = np.abs(total_pos - total_neg) / denom
            # only IDR positions that are themselves charged get the weighting
            idr_charged = np.array([1.0 if r in CHARGED else 0.0 for r in idr])
            repulsive_mask = charge_weight * idr_charged
            w = w - (w * repulsive_mask * prefactor)

        if use_aliphatic_weighting:
            ali_weights = np.array(
                [self._ali_weight(s_level, int(idr_ali[j])) for j in range(len(idr))]
            )
            w = w * ali_weights

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
            in surface order than ``key_b``) to the baseline-subtracted interaction
            value (negative = attractive).
        """
        surf = self.surface_residues
        imc = self.IMC_object
        baseline = imc.null_interaction_baseline
        prefactor = imc.charge_prefactor
        lookup = imc.lookup

        # precompute patch context per surface residue
        ctx = {}
        for rec in surf:
            n_pos, n_neg, _ = self._patch_charge_counts(rec.key)
            ctx[rec.key] = (n_pos, n_neg, self._patch_aliphatic_level(rec.key))

        results = {}
        for i in range(len(surf)):
            for j in range(i + 1, len(surf)):
                ra, rb = surf[i], surf[j]
                aa, ab = ra.one_letter, rb.one_letter
                val = float(lookup[aa][ab])

                if (
                    use_charge_weighting
                    and prefactor is not None
                    and aa in CHARGED
                    and ab in CHARGED
                ):
                    pa, na, la = ctx[ra.key]
                    pb, nb, lb = ctx[rb.key]
                    total_pos = pa + pb
                    total_neg = na + nb
                    total_charge = total_pos + total_neg
                    cw = abs(total_pos - total_neg) / (
                        total_charge if total_charge else 1.0
                    )
                    val = val - (val * cw * prefactor)

                if use_aliphatic_weighting:
                    la = ctx[ra.key][2]
                    lb = ctx[rb.key][2]
                    val = val * self._ali_weight(la, lb)

                results[(ra.key, rb.key)] = val - baseline
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

        A surface residue is reachable if its distance from the anchor is no greater
        than the reach of the IDR's final residue, ``R(idr_length)``. For each
        reachable residue the smallest IDR index able to reach it is also returned.

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
        anchor_pos = self._anchor_position(anchor)
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

    def write_pdb_IDR_interaction(
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
