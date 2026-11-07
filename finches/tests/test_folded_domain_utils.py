"""
Tests for finches.utils.folded_domain_utils.FoldedDomain (the patch mode).
"""

import os

import numpy as np
import pytest

from finches.utils.folded_domain_utils import FoldedDomain, FoldeDomain

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDB = os.path.normpath(
    os.path.join(_HERE, "..", "..", "examples", "docs_demo", "ADBD1.pdb")
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(_PDB), reason="example PDB not available"
)

_IDR = "MSEYQPSLFALNPMGFSPLDGSKSTNENVSASTSTAKPMVGQLIFDKFIKTEEDPIIKQDTPSNLDFD"


@pytest.fixture(scope="module")
def imc():
    from finches.frontend.mpipi_frontend import Mpipi_frontend

    return Mpipi_frontend().IMC_object


def test_legacy_alias():
    # the original misspelt class name still works
    assert FoldeDomain is FoldedDomain


def test_surface_classification():
    fd = FoldedDomain(_PDB)
    assert len(fd) == len(fd.sequence) == len(fd.sasa) == len(fd.surface_vector)
    assert fd.surface_indices == [i for i, v in enumerate(fd.surface_vector) if v]
    assert set(fd.surface_positions) == set(fd.surface_indices)


def test_neighbours_include_self():
    fd = FoldedDomain(_PDB)
    for idx, neighbours in fd.surface_neighbours.items():
        assert idx in [n for n, _ in neighbours]


def test_patch_intermap_does_not_change_the_neighbour_graph(imc):
    # the patch intermap uses its own patch_radius, and must not leave the object's
    # neighbour graph rebuilt at that radius (surface epsilons used to depend on
    # whether the intermap had been computed first)
    fd = FoldedDomain(_PDB)
    before = {k: v[2] for k, v in fd.calculate_surface_epsilon(_IDR, imc).items()}
    fd.calculate_idr_surface_patch_interactions(_IDR, imc, 31, patch_radius=12)
    after = {k: v[2] for k, v in fd.calculate_surface_epsilon(_IDR, imc).items()}
    assert before.keys() == after.keys()
    assert all(before[k] == after[k] for k in before)


def test_patch_intermap_shape_and_buried_rows(imc):
    fd = FoldedDomain(_PDB)
    out = fd.calculate_idr_surface_patch_interactions(_IDR, imc, 31)
    X = out[1]
    assert X.shape == (len(fd), len(_IDR) - 31 + 1)
    surface = set(fd.surface_indices)
    for i in range(len(fd)):
        if i in surface:
            assert np.all(np.isfinite(X[i]))
        else:
            assert np.all(np.isnan(X[i]))

    with pytest.raises(ValueError):
        fd.calculate_idr_surface_patch_interactions(_IDR, imc, 30)
    with pytest.raises(ValueError):
        fd.calculate_idr_surface_patch_interactions("GSGS", imc, 31)


def test_rebuilding_the_graph_refreshes_cached_paths():
    fd = FoldedDomain(_PDB)
    paths = fd.all_shortest_paths
    fd.get_nearest_neighbour_res(distance_thresh=12.0)
    assert fd.all_shortest_paths is not paths


def test_iwd_on_a_disconnected_surface_graph():
    # residues in different connected components have no over-the-surface path;
    # they contribute zero rather than raising a KeyError
    fd = FoldedDomain(_PDB)
    fd.get_nearest_neighbour_res(distance_thresh=4.0)
    iwd, n, null = fd.calculate_IWD(["K", "R", "E", "D"])
    assert np.isfinite(iwd) and iwd >= 0
    assert n >= 2 and null == -1

    # fewer than two selected residues gives a list of Nones
    assert fd.calculate_IWD(["W"], positions=[]) == [None, None, None]

    with pytest.raises(ValueError):
        fd.calculate_IWD(["K"], distance_mode="nonsense")


def test_attractive_and_repulsive_partition(imc):
    fd = FoldedDomain(_PDB)
    values = [v[2] for v in fd.calculate_surface_epsilon(_IDR, imc).values()]
    att = fd.calculate_attractive_surface_epsilon(_IDR, imc)
    rep = fd.calculate_repulsive_surface_epsilon(_IDR, imc)
    assert len(att) + len(rep) == len(values)
    assert all(v < 0 for v in att) and all(v >= 0 for v in rep)


def test_start_end_requires_ignore_warnings():
    with pytest.raises(ValueError):
        FoldedDomain(_PDB, start=0, end=40)
