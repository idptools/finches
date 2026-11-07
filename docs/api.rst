Frontend Object API
=====================

The **frontend objects** are the main user-facing entry point to FINCHES. Almost
everything you will want to do — computing epsilon values, building interaction
maps (intermaps), per-residue interaction profiles, phase diagrams, and deep
mutational scans — is exposed as a method on one of two objects:

* :class:`~finches.frontend.mpipi_frontend.Mpipi_frontend` — the Mpipi-GG model.
* :class:`~finches.frontend.calvados_frontend.CALVADOS_frontend` — the CALVADOS2 model.

Both objects inherit from a common base class
(:class:`~finches.frontend.frontend_base.FinchesFrontend`), which implements every
method, so they expose an **identical interface**: code written against one frontend
will work against the other simply by swapping the object you instantiate. The two
differ only in the underlying coarse-grained forcefield and in one class attribute,
``default_intermap_limit``, the colour limit ``interaction_figure()`` uses when you
do not set ``vmin`` and ``vmax`` yourself.

.. contents:: On this page
   :local:
   :depth: 2


Choosing a frontend
-------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Capability
     - ``Mpipi_frontend``
     - ``CALVADOS_frontend``
   * - Forcefield
     - Mpipi-GG (Joseph *et al.*, 2021)
     - CALVADOS2 (Tesei & Lindorff-Larsen, 2022)
   * - Protein–protein interactions
     - Yes
     - Yes
   * - Solution conditions
     - ``salt``, ``dielectric``
     - ``salt``, ``pH``, ``temperature``
   * - Default intermap colour limits (``default_intermap_limit``)
     - -3 to +3
     - -7.5 to +7.5

If you are unsure which to use, ``Mpipi_frontend`` is the most broadly applicable
starting point and is the model used for most published FINCHES analyses. Use
``CALVADOS_frontend`` when you specifically want CALVADOS2 physics or to vary pH.

.. note::

   The two models are calibrated independently, so **epsilon values are not
   directly comparable between forcefields** — only compare values computed with
   the same frontend. See :doc:`epsilon` and :doc:`general_caveats` for details.


Quickstart
----------

.. code-block:: python

    # both frontends are importable directly from the top-level package
    from finches import Mpipi_frontend, CALVADOS_frontend

    # instantiate with default solution conditions
    mf = Mpipi_frontend()                        # salt=0.15 M, dielectric=80
    cf = CALVADOS_frontend()                      # salt=0.15 M, pH=7.4, T=288 K

    # the LAF-1 RGG domain and C-terminal IDR
    seq1 = "MESNQSNNGGSGNAALNRGGRYVPPHLRGGDGGAAAAASAGGDDRRGGAGGGGYRRGGGNSGGGGGGGYDRGYNDNRDDRDNRGGSGGYGRDRNYEDRGYNGGGGGGGNRGYNNNRGGGGGGYNRQDRGDGGSSNFSRGGYNNRDEGSDNRGSGRSYNNDRRDNGGD"
    seq2 = "LEGMSGDMRSGGGYRGRGGRGNGQRFGGRDHRYQGGSGNGGGGNGGGGGFGGGGQRSGGGGGFQSGGGGGRQQQQQQRAQPQQDWWS"

    # a single mean-field interaction parameter (negative = attractive)
    eps = mf.epsilon(seq1, seq2)

    # a publication-ready interaction map (intermap)
    fig, im, ax_main, ax_top, ax_right, ax_cbar = mf.interaction_figure(seq1, seq2)

    # a homotypic phase diagram
    data, fig, ax = mf.plot_phase_diagram(seq1)

Solution conditions are set at construction time, e.g.
``Mpipi_frontend(salt=0.05, dielectric=80.0)`` or
``CALVADOS_frontend(salt=0.15, pH=5.5, temperature=310)``.

The sliding-window methods (the intermap, the per-residue profiles and the
protein–peptide profiles) need sequences at least as long as their
window, and the profile methods smooth with a 30-point Savitzky-Golay filter by
default, so they need a profile of at least 30 windows; pass
``smoothing_window=None`` to turn smoothing off for short sequences. Invalid
arguments raise ``ValueError`` throughout.


The interface at a glance
-------------------------

Every method below is available on **both** frontends (unless noted). They are
grouped here by task; full signatures and detailed descriptions follow in the
per-object reference sections.

**Mean-field interaction strength (epsilon)**

* :meth:`~finches.frontend.frontend_base.FinchesFrontend.epsilon` — single scalar
  interaction parameter between two sequences.
* :meth:`~finches.frontend.frontend_base.FinchesFrontend.epsilon_vectors` —
  per-residue attractive and repulsive components of epsilon.

**Interaction maps (intermaps)**

* :meth:`~finches.frontend.frontend_base.FinchesFrontend.intermolecular_idr_matrix`
  — the raw sliding-window interaction matrix plus disorder profiles.
* :meth:`~finches.frontend.frontend_base.FinchesFrontend.interaction_figure` — a
  publication-ready heatmap of that matrix with disorder tracks.

**Per-residue interaction profiles**

* :meth:`~finches.frontend.frontend_base.FinchesFrontend.per_residue_attractive_vector`
  / :meth:`~finches.frontend.frontend_base.FinchesFrontend.per_residue_repulsive_vector`
  — sticker / spacer profiles for one sequence against another.
* :meth:`~finches.frontend.frontend_base.FinchesFrontend.protein_peptide_vector`
  / :meth:`~finches.frontend.frontend_base.FinchesFrontend.plot_protein_peptide_vector`
  — binding propensity along a protein against a chosen peptide.

**Phase diagrams**

* :meth:`~finches.frontend.frontend_base.FinchesFrontend.build_phase_diagram` —
  Flory–Huggins binodal/spinodal data for a homotypic system.
* :meth:`~finches.frontend.frontend_base.FinchesFrontend.plot_phase_diagram`
  / :meth:`~finches.frontend.frontend_base.FinchesFrontend.plot_multiple_phase_diagrams`
  — plot one or several phase diagrams.

**Deep mutational scanning (DMS)**

* :meth:`~finches.frontend.frontend_base.FinchesFrontend.dms` — homotypic epsilon
  for every single-point mutant.
* :meth:`~finches.frontend.frontend_base.FinchesFrontend.plot_dms` — heatmap of a
  DMS scan.


.. _mpipi-frontend-reference:

Mpipi_frontend
--------------

``Mpipi_frontend`` uses the Mpipi-GG forcefield, with the salt concentration and
dielectric constant as its solution conditions. Disorder prediction (the
``disorder_1``/``disorder_2`` arguments, which default to ``None``) runs with
metapredict, as does the masking of folded domains with ``zero_folded``.

.. autoclass:: finches.frontend.mpipi_frontend.Mpipi_frontend
   :members:
   :exclude-members: protein_nucleic_vector, plot_protein_nucleic_vector, supports_rna
   :inherited-members:
   :show-inheritance:


.. _calvados-frontend-reference:

CALVADOS_frontend
-----------------

``CALVADOS_frontend`` uses the CALVADOS2 forcefield and additionally lets you set the
solution ``pH`` and ``temperature``.

.. autoclass:: finches.frontend.calvados_frontend.CALVADOS_frontend
   :members:
   :exclude-members: protein_nucleic_vector, plot_protein_nucleic_vector, supports_rna
   :inherited-members:
   :show-inheritance:


Shared base class
-----------------

Both frontends inherit from ``FinchesFrontend``. It is documented here for
completeness and should **not** be instantiated directly — always use one of the
derived classes above.

.. autoclass:: finches.frontend.frontend_base.FinchesFrontend
   :members:
   :exclude-members: protein_nucleic_vector, plot_protein_nucleic_vector, supports_rna
   :show-inheritance:


Forcefield models
-----------------

The frontends are built on a forcefield model, which supplies the residue-level pair potentials and integrates them into the interaction parameters FINCHES uses. You only need these classes to inspect the underlying potentials, to change solution conditions on an existing object, or to add a new forcefield (see ``finches/forcefields/readme.md``). Both models implement ``compute_full_potential()`` and ``compute_interaction_parameter()`` with the same signatures; the older model-specific names (``compute_full_Mpipi()``, ``compute_full_calvados()``, ``compute_colulomb()``) still work.

.. autoclass:: finches.forcefields.mpipi.Mpipi_model
   :members:
   :show-inheritance:

.. autoclass:: finches.forcefields.calvados.CALVADOS_model
   :members:
   :show-inheritance:

.. autoclass:: finches.forcefields.model_base.ForcefieldModel
   :members:


Structure-surface analysis
--------------------------

``StructureSurface`` (in ``finches.utils.structure_surface``) is the structure-aware mode for IDR:folded domain intermaps described in :doc:`idr_fd`, and the one we recommend. It takes a frontend at construction, decomposes the structure into folded domains and IDRs, classifies the folded surface, builds the contiguous-surface net, and exposes the scoring methods, the intermap and its plotting helper, and the PDB writers.

.. autoclass:: finches.utils.structure_surface.StructureSurface
   :members:

.. autofunction:: finches.utils.structure_surface.plot_surface_intermap

The residue records the class hands back:

.. autoclass:: finches.utils.structure_surface.Residue
   :members:


Folded-domain utilities
-----------------------

The original patch mode for IDR:folded domain maps lives in ``FoldedDomain``. It is superseded by ``StructureSurface`` and kept so that earlier results can be reproduced; :doc:`idr_fd` explains both modes and how they differ.

.. autoclass:: finches.utils.folded_domain_utils.FoldedDomain
   :members:

.. autofunction:: finches.utils.folded_domain_utils.extract_and_write_domains
