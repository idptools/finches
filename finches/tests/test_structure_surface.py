"""
Tests for finches.utils.structure_surface.StructureSurface.
"""

import os

import numpy as np
import pytest

from finches.utils.structure_surface import (
    StructureSurface,
    DEFAULT_REACH_B,
    DEFAULT_REACH_NU,
)

# locate the example PDB shipped with the repo
_HERE = os.path.dirname(os.path.abspath(__file__))
_PDB = os.path.normpath(
    os.path.join(_HERE, "..", "..", "examples", "docs_demo", "ADBD1.pdb")
)


pytestmark = pytest.mark.skipif(
    not os.path.exists(_PDB), reason="example PDB not available"
)


@pytest.fixture(scope="module")
def frontend():
    # heavier import kept inside the fixture so collection stays cheap
    from finches.frontend.mpipi_frontend import Mpipi_frontend

    return Mpipi_frontend()


@pytest.fixture(scope="module")
def ss(frontend):
    return StructureSurface(_PDB, frontend)


# ----------------------------------------------------------------------
# decomposition (capability 1) + surface classification (capability 2)
# ----------------------------------------------------------------------
def test_decomposition_invariants(ss):
    surf = {r.key for r in ss.surface_residues}
    fold = {r.key for r in ss.folded_residues}
    modeled = {r.key for r in ss.residues if r.modeled}
    idr = {r.key for r in ss.idr_residues}

    assert surf <= fold <= modeled
    assert surf.isdisjoint(idr)
    # every residue is exactly one of folded or idr
    assert {r.key for r in ss.residues} == fold | idr
    assert fold.isdisjoint(idr)


def test_idr_segments_have_anchor(ss):
    assert len(ss.idr_segments) >= 1
    for seg in ss.idr_segments:
        assert len(seg["residues"]) >= 1
        # a terminal IDR has exactly one folded anchor; internal loops up to two
        anchors = [a for a in (seg["n_anchor"], seg["c_anchor"]) if a is not None]
        assert len(anchors) >= 1
        for a in anchors:
            assert a.domain == "folded" and a.modeled


def test_surface_positions_present(ss):
    for r in ss.surface_residues:
        assert r.position is not None
        assert r.position.shape == (3,)


# ----------------------------------------------------------------------
# missing residues via provided sequence
# ----------------------------------------------------------------------
def test_missing_residues_become_idr(frontend):
    base = StructureSurface(_PDB, frontend)
    # full chain sequence = modeled sequence + a disordered C-terminal extension
    modeled_seq = "".join(
        r.one_letter
        for r in sorted(base.residues, key=lambda x: x.seq_index)
        if r.modeled
    )
    extension = "GSGSGSPRGRGRGS"
    full = modeled_seq + extension

    ss = StructureSurface(_PDB, frontend, sequences=full)
    missing = [r for r in ss.residues if not r.modeled]
    assert len(missing) == len(extension)
    # all missing residues are IDR and never surface
    assert all(r.domain == "idr" for r in missing)
    assert all(not r.surface for r in missing)


def test_sequence_mismatch_raises(frontend):
    base = StructureSurface(_PDB, frontend)
    n = len([r for r in base.residues if r.modeled])
    # a clearly wrong sequence (all tryptophan) should fail identity validation
    with pytest.raises(ValueError):
        StructureSurface(_PDB, frontend, sequences="W" * n)


def test_constructor_requires_frontend():
    # a bad "frontend" (no IMC_object / lookup) must raise TypeError
    with pytest.raises(TypeError):
        StructureSurface(_PDB, object())


# ----------------------------------------------------------------------
# contiguous-surface net (capability 3)
# ----------------------------------------------------------------------
def test_surface_net_basic(ss):
    g = ss.surface_graph
    assert g.number_of_nodes() == len(ss.surface_residues)
    # neighbours are symmetric and only between surface residues
    surf_keys = {r.key for r in ss.surface_residues}
    for k, nbrs in ss.surface_neighbours.items():
        assert k in surf_keys
        for nb in nbrs:
            assert nb in surf_keys
            assert k in ss.surface_neighbours[nb]


def test_occlusion_prunes_through_core(frontend):
    # an edge whose midpoint is surrounded by atoms on all sides must be rejected,
    # while an edge over open space (atoms only on one side) is kept
    ss = StructureSurface(_PDB, frontend)
    from scipy.spatial import cKDTree

    # spherical shell of atoms centred at the origin -> any chord through it has a
    # surrounded midpoint
    rng = np.random.default_rng(0)
    v = rng.normal(size=(400, 3))
    shell = 8.0 * v / np.linalg.norm(v, axis=1)[:, None]
    ss._folded_heavy_xyz = shell
    tree = cKDTree(shell)
    # chord through the centre -> pruned
    assert (
        ss._segment_on_surface(np.array([-8.0, 0, 0]), np.array([8.0, 0, 0]), tree)
        is False
    )

    # a segment far outside the shell (open space, few/one-sided atoms) -> kept
    assert (
        ss._segment_on_surface(np.array([20.0, 0, 0]), np.array([20.0, 8.0, 0]), tree)
        is True
    )


def test_surface_patch_includes_self(ss):
    for r in ss.surface_residues[:5]:
        patch = ss.surface_patch(r.key)
        assert r.key in patch


# ----------------------------------------------------------------------
# reach model (capability 5)
# ----------------------------------------------------------------------
def test_reach_radius_monotonic(ss):
    n = np.arange(1, 50)
    R = ss.reach_radius(n)
    assert np.all(np.diff(R) > 0)
    assert ss.reach_radius(1) == pytest.approx(DEFAULT_REACH_B)
    assert ss.reach_radius(10) == pytest.approx(DEFAULT_REACH_B * 10**DEFAULT_REACH_NU)


def test_reachable_grows_with_length(ss):
    anchor = ss.surface_residues[0].key
    short = ss.reachable_surface_residues(anchor, idr_length=5)
    long = ss.reachable_surface_residues(anchor, idr_length=80)
    assert set(short) <= set(long)
    assert len(long) >= len(short)
    # the anchor's own neighbourhood should be reachable even by a short IDR
    assert len(short) >= 1


# ----------------------------------------------------------------------
# context-aware interaction scoring (capability 4)
# ----------------------------------------------------------------------
def test_surface_vs_idr_returns_scores(ss):
    res = ss.surface_vs_idr("GSGSGSGSGS")
    assert set(res) == {r.key for r in ss.surface_residues}
    for v in res.values():
        assert np.isfinite(v["score"])
        assert v["n_reachable"] == len("GSGSGSGSGS")


def test_surface_vs_idr_charge_response(ss):
    # poly-R and poly-E challenge the same surface differently
    rr = ss.surface_vs_idr("RRRRRRRRRR")
    ee = ss.surface_vs_idr("EEEEEEEEEE")
    diff = np.array([rr[k]["score"] - ee[k]["score"] for k in rr])
    assert np.any(np.abs(diff) > 1e-6)


def test_respect_reach_gating(ss):
    anchor = ss.surface_residues[0].key
    gated = ss.surface_vs_idr("RGRGRGRGRG", anchor=anchor, respect_reach=True)
    nreach = np.array([v["n_reachable"] for v in gated.values()])
    # the anchor residue itself is fully reachable; distal residues less so
    assert nreach.max() == len("RGRGRGRGRG")
    assert nreach.min() < nreach.max()

    with pytest.raises(ValueError):
        ss.surface_vs_idr("RG", respect_reach=True)  # no anchor


def test_contour_separation_direction(ss):
    n = ss._contour_separation(5, "N")
    c = ss._contour_separation(5, "C")
    assert list(n) == [1, 2, 3, 4, 5]
    assert list(c) == [5, 4, 3, 2, 1]

    with pytest.raises(ValueError):
        ss._contour_separation(5, "X")


def test_idr_tether_resolves_anchor(ss):
    seg = next(
        s
        for s in ss.idr_segments
        if s["n_anchor"] is not None or s["c_anchor"] is not None
    )
    anchor, tether = ss.idr_tether(seg)

    assert tether in ("N", "C")
    assert anchor in {r.key for r in ss.folded_residues}
    # a C-terminal anchor means the IDR runs off the N-side, so its C-terminus is tethered
    if seg["c_anchor"] is not None:
        assert tether == "C" and anchor == seg["c_anchor"].key
    else:
        assert tether == "N" and anchor == seg["n_anchor"].key

    # an index into idr_segments resolves the same way
    idx = ss.idr_segments.index(seg)
    assert ss.idr_tether(idx) == (anchor, tether)


def test_surface_vs_idr_matrix_shape(ss):
    seq = "GSGSGSGSGSRGRGRGRGRG"
    window = 5
    X, keys, centres = ss.surface_vs_idr_matrix(seq, window_size=window)

    assert X.shape == (len(ss.surface_residues), len(seq) - window + 1)
    assert keys == [r.key for r in ss.surface_residues]
    # each column is centred on its window, so centres start half a window in
    assert centres[0] == (window - 1) // 2
    assert centres[-1] == len(seq) - 1 - (window - 1) // 2
    assert np.all(np.isfinite(X))


def test_surface_vs_idr_matrix_cis_gating(ss):
    seq = "RGRGRGRGRGRGRGRGRGRGRGRGR"
    anchor, tether = ss.idr_tether(0)

    trans, _, _ = ss.surface_vs_idr_matrix(seq, window_size=5)
    # a very short reach means most surface residues can barely be contacted, so cells
    # decay towards zero rather than being cut off at a hard boundary
    tight, _, _ = ss.surface_vs_idr_matrix(
        seq, window_size=5, anchor=anchor, tether=tether, reach_b=0.5
    )
    # note a single cell can grow: it is a signed sum, so damping one contribution can
    # unmask another that was cancelling it. The aggregate is what has to collapse.
    assert np.mean(np.abs(tight)) < 0.05 * np.mean(np.abs(trans))

    # a reach long enough to cover the whole domain has to give trans back. reach_b sets
    # the footprint as well as the tether, so the control has to use the same value.
    huge = 1e6
    loose, _, _ = ss.surface_vs_idr_matrix(
        seq, window_size=5, anchor=anchor, tether=tether, reach_b=huge
    )
    trans_loose, _, _ = ss.surface_vs_idr_matrix(seq, window_size=5, reach_b=huge)
    assert np.allclose(loose, trans_loose, atol=1e-6)

    # which terminus is tethered changes which IDR residues are close to the surface
    from_n, _, _ = ss.surface_vs_idr_matrix(
        seq, window_size=5, anchor=anchor, tether="N", reach_b=1.5
    )
    from_c, _, _ = ss.surface_vs_idr_matrix(
        seq, window_size=5, anchor=anchor, tether="C", reach_b=1.5
    )
    assert not np.allclose(from_n, from_c)


def test_surface_vs_idr_matrix_rows_all(ss):
    seq = "GSGSGRGRGSSEEDSGSGSG"
    surf_X, surf_keys, _ = ss.surface_vs_idr_matrix(seq, window_size=5)
    all_X, all_keys, _ = ss.surface_vs_idr_matrix(seq, window_size=5, rows="all")

    # one row per residue, in chain/sequence order
    assert all_X.shape[0] == len(ss.residues)
    ordered = sorted(ss.residues, key=lambda r: (r.chain_index, r.seq_index))
    assert all_keys == [r.key for r in ordered]

    row_of = {k: i for i, k in enumerate(all_keys)}

    # buried folded residues are nan (so they can be greyed out), everything else scored
    for rec in ordered:
        row = all_X[row_of[rec.key]]
        if rec.domain == "folded" and not rec.surface:
            assert np.all(np.isnan(row))
        else:
            assert np.all(np.isfinite(row))

    # IDR residues do get scored
    assert any(r.domain == "idr" for r in ordered)

    # surface rows are untouched by the row selection
    for i, k in enumerate(surf_keys):
        assert np.allclose(surf_X[i], all_X[row_of[k]])


def test_idr_residue_context_is_linear(ss):
    # a surface residue reads its context from its 3D patch, an IDR residue from the
    # linear sequence -- the two paths must both resolve without a patch lookup
    surf_rec = ss.surface_residues[0]
    idr_rec = next(r for r in ss.idr_residues if r.modeled)

    assert (
        ss._residue_context(surf_rec)[:2] == ss._patch_charge_counts(surf_rec.key)[:2]
    )

    pos_w, neg_w, ali = ss._chain_linear_context(idr_rec.chain_index)
    n_pos, n_neg, level = ss._residue_context(idr_rec)
    assert n_pos == pytest.approx(pos_w[idr_rec.seq_index])
    assert n_neg == pytest.approx(neg_w[idr_rec.seq_index])
    assert level == ali[idr_rec.seq_index]


def test_surface_vs_idr_matrix_validation(ss):
    with pytest.raises(ValueError):
        ss.surface_vs_idr_matrix("GSGSGSGS", window_size=4)  # even window

    with pytest.raises(ValueError):
        ss.surface_vs_idr_matrix("GSGS", window_size=31)  # window longer than IDR

    with pytest.raises(ValueError):
        ss.surface_vs_idr_matrix("GSGSGSGS", window_size=5, tether="C")  # no anchor

    with pytest.raises(ValueError):
        ss.surface_vs_idr_matrix("GSGSGSGS", window_size=5, rows="nonsense")

    with pytest.raises(ValueError):
        ss.surface_vs_idr_matrix(
            "GSGSGSGS", window_size=5, anchor=ss.surface_residues[0].key, tether="X"
        )

    # an empty IDR gives an empty map rather than raising
    X, keys, centres = ss.surface_vs_idr_matrix("", window_size=5)
    assert X.size == 0 and keys == [] and centres.size == 0


def test_surface_vs_surface(ss):
    res = ss.surface_vs_surface()
    n = len(ss.surface_residues)
    assert len(res) == n * (n - 1) // 2
    for v in res.values():
        assert np.isfinite(v)


# ----------------------------------------------------------------------
# PDB output (beta-column writers)
# ----------------------------------------------------------------------
def _beta_by_resseq(path):
    """Read the per-residue beta (B-factor) column back from a PDB file."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                out[int(line[22:26])] = float(line[60:66])
    return out


def test_write_pdb_solvent_accessibility(ss, tmp_path):
    out = ss.write_pdb_solvent_accessibility(str(tmp_path / "sa.pdb"))
    beta = _beta_by_resseq(out)
    # every modeled residue is written and beta is binary
    modeled = [r for r in ss.residues if r.modeled]
    assert len(beta) == len({r.res_seq for r in modeled})
    assert set(beta.values()) <= {0.0, 1.0}
    # surface and IDR residues are accessible (1); buried folded are 0
    for r in ss.surface_residues:
        assert beta[r.res_seq] == 1.0
    for r in ss.idr_residues:
        if r.modeled:
            assert beta[r.res_seq] == 1.0
    buried = [r for r in ss.folded_residues if not r.surface]
    for r in buried:
        assert beta[r.res_seq] == 0.0


def test_write_pdb_groups(ss, tmp_path):
    key = ss.surface_residues[len(ss.surface_residues) // 2].key
    out = ss.write_pdb_groups(key, str(tmp_path / "grp.pdb"), include_self=True)
    beta = _beta_by_resseq(out)
    flagged = {rs for rs, b in beta.items() if b == 1.0}
    expected = {key[1]} | {nb[1] for nb in ss.surface_neighbours[key]}
    assert flagged == expected

    # a non-surface residue has no surface net and must raise
    idr_key = next(r.key for r in ss.idr_residues if r.modeled)
    with pytest.raises(ValueError):
        ss.write_pdb_groups(idr_key, str(tmp_path / "bad.pdb"))


def test_write_pdb_idr_interaction(ss, tmp_path):
    out = ss.write_pdb_IDR_interaction("RRKRRKRRKRRK", str(tmp_path / "idr.pdb"))
    beta = _beta_by_resseq(out)
    scores = ss.surface_vs_idr("RRKRRKRRKRRK")
    # surface residues carry their (finite) interaction score
    for key, info in scores.items():
        assert beta[key[1]] == pytest.approx(info["score"], abs=1e-2)
    # at least one attractive (negative) value made it into the beta column
    assert min(beta.values()) < 0.0


def test_write_pdb_accessible_from_position(ss, tmp_path):
    idr_key = next(r.key for r in ss.idr_residues if r.modeled)
    out = ss.write_pdb_accessible_from_position(idr_key, str(tmp_path / "acc.pdb"))
    beta = _beta_by_resseq(out)
    flagged = {rs for rs, b in beta.items() if b == 1.0}
    # only surface residues can be flagged reachable
    surface_rs = {r.res_seq for r in ss.surface_residues}
    assert flagged <= surface_rs

    # a folded-domain position is not allowed
    folded_key = ss.folded_residues[0].key
    with pytest.raises(ValueError):
        ss.write_pdb_accessible_from_position(folded_key, str(tmp_path / "bad.pdb"))


def test_matrix_units_match_finches_epsilon(ss):
    """ACCEPTANCE: the aggregation must reproduce a FINCHES epsilon exactly.

    With the weighting terms off (so the deliberately-3D surface context cannot
    differ from FINCHES' linear one) a patch-only cell must equal
    calculate_epsilon_value(window_sequence, patch_sequence) to machine precision.
    This is what makes the values comparable with an IDR:IDR intermap.
    """
    seq = "GSGSGRGRGSSEEDSGSGSGRGRGSSEEDS"
    W = 11
    imc = ss.IMC_object
    surf_keys = {r.key for r in ss.surface_residues}

    X, keys, _ = ss.surface_vs_idr_matrix(
        seq,
        window_size=W,
        use_footprint=False,
        use_charge_weighting=False,
        use_aliphatic_weighting=False,
    )

    for i in (0, len(keys) // 2, len(keys) - 1):
        patch_seq = "".join(
            ss.get_residue(*k).one_letter
            for k in ss.surface_patch(keys[i])
            if k in surf_keys
        )
        for c in (0, len(seq) - W):
            ref = imc.calculate_epsilon_value(
                seq[c : c + W],
                patch_seq,
                use_charge_weighting=False,
                use_aliphatic_weighting=False,
            )
            assert X[i, c] == pytest.approx(ref, abs=1e-10)


def test_footprint_grows_with_window(ss):
    # the footprint radius is R(window//2), so a longer window reaches further
    key = ss.surface_residues[0].key
    order, dists = ss._footprint_order(key)

    assert dists[0] == pytest.approx(0.0)  # the centre is its own nearest
    assert np.all(np.diff(dists) >= 0)  # sorted by distance

    sizes = [
        int(np.searchsorted(dists, ss.reach_radius(W // 2), side="right"))
        for W in (5, 11, 21, 31)
    ]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_footprint_changes_the_map(ss):
    seq = "GSGSGRGRGSSEEDSGSGSGRGRGSSEEDS"
    patch_only, keys, _ = ss.surface_vs_idr_matrix(
        seq, window_size=11, use_footprint=False
    )
    footprint, _, _ = ss.surface_vs_idr_matrix(seq, window_size=11, use_footprint=True)
    assert not np.allclose(patch_only, footprint)
    assert np.all(np.isfinite(footprint))


def test_footprint_environment_matters(ss):
    # residues of the same type in different surface environments must differ
    seq = "GSGSGRGRGSSEEDSGSGSGRGRGSSEEDS"
    X, keys, _ = ss.surface_vs_idr_matrix(seq, window_size=11)
    aas = [ss.get_residue(*k).one_letter for k in keys]

    repeated = [a for a in set(aas) if aas.count(a) > 2]
    assert repeated
    idx = [i for i, a in enumerate(aas) if a == repeated[0]]
    assert np.ptp(X[idx].mean(axis=1)) > 1e-6


# a real IDR sequence: a repetitive one barely varies window to window, which makes
# any row-similarity measure read mostly noise
_SMOOTH_SEQ = (
    "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWQPEVKLDLDTASSQLADDVYEVVLRVTVTASLGEETAFLCEVQQ"
)


def _row_disagreement(ss, X, lo, hi):
    """Mean |row_a - row_b| over surface pairs with geodesic distance in [lo, hi)."""
    surf = ss.surface_residues
    sd = ss.surface_distance
    out = []
    for a in range(len(surf)):
        for b in range(a + 1, len(surf)):
            d = sd.get(surf[a].key, {}).get(surf[b].key, None)
            if d is not None and lo <= d < hi:
                out.append(np.mean(np.abs(X[a] - X[b])))
    return float(np.mean(out)), len(out)


def test_map_is_spatially_smooth(ss):
    # neighbouring surface residues see nearly the same window, so their rows must
    # agree far better than rows for residues on opposite sides of the domain
    X, _, _ = ss.surface_vs_idr_matrix(_SMOOTH_SEQ, window_size=31)

    near, n_near = _row_disagreement(ss, X, 0.0, 8.0)
    far, n_far = _row_disagreement(ss, X, 8.0, np.inf)

    assert n_near > 10 and n_far > 10
    assert near < 0.5 * far


def test_contact_radius_controls_smoothness(ss):
    # the contact shell is what stops an IDR residue behaving as a point contact, so
    # widening it has to make touching neighbours disagree less
    ratios = []
    for radius in (0.0, 3.0, 6.0, 9.0):
        X, _, _ = ss.surface_vs_idr_matrix(
            _SMOOTH_SEQ, window_size=31, contact_radius=radius
        )
        assert np.all(np.isfinite(X))
        near = _row_disagreement(ss, X, 0.0, 8.0)[0]
        far = _row_disagreement(ss, X, 8.0, np.inf)[0]
        ratios.append(near / far)

    assert ratios == sorted(ratios, reverse=True)
    assert ratios[-1] < ratios[0]


def test_contact_radius_preserves_centre_residue_signal(ss):
    # smoothing must not wash out the identity of the residue the row is about:
    # chemically identical residues in different environments still have to differ
    X, keys, _ = ss.surface_vs_idr_matrix(_SMOOTH_SEQ, window_size=31)
    aas = [ss.get_residue(*k).one_letter for k in keys]

    repeated = [a for a in set(aas) if aas.count(a) > 2]
    assert repeated
    idx = [i for i, a in enumerate(aas) if a == repeated[0]]
    assert np.ptp(X[idx].mean(axis=1)) > 0.1 * np.std(X)


def test_contact_radius_validation(ss):
    with pytest.raises(ValueError, match="contact_radius"):
        ss.surface_vs_idr_matrix("GSGSGSGSGS", window_size=5, contact_radius=-1.0)


def test_cis_gating_decays_smoothly_with_anchor_distance(ss):
    # the tether reach is a mean end-to-end distance, not a hard limit, so cis has to
    # fade out with distance from the anchor rather than stopping at a boundary
    seq = "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWQPEVKLDLDTASSQLADDVYEVVLRVTVTASLGEETAFLC"
    anchor, tether = ss.idr_tether(0)

    trans, _, _ = ss.surface_vs_idr_matrix(seq, window_size=11)
    cis, _, _ = ss.surface_vs_idr_matrix(
        seq, window_size=11, anchor=anchor, tether=tether, reach_b=1.0
    )

    apos = ss._anchor_position(anchor)
    dist = np.array([np.linalg.norm(apos - r.position) for r in ss.surface_residues])

    # retained signal, as a fraction of trans, must fall off as you move away
    keep = np.mean(np.abs(cis), axis=1) / np.maximum(
        np.mean(np.abs(trans), axis=1), 1e-12
    )
    assert np.corrcoef(dist, keep)[0, 1] < -0.5

    # and it fades rather than switching off: with a partly-covered domain there are
    # residues sitting at intermediate attenuation, which a hard cutoff cannot produce
    assert np.sum((keep > 0.01) & (keep < 0.9)) > 3
