Frontend Object API
=====================

The **frontend objects** are the main user-facing entry point to FINCHES. Almost
everything you will want to do — computing epsilon values, building interaction
maps (intermaps), per-residue interaction profiles, phase diagrams, and deep
mutational scans — is exposed as a method on one of two objects:

* :class:`~finches.frontend.mpipi_frontend.Mpipi_frontend` — the Mpipi-GG model.
* :class:`~finches.frontend.calvados_frontend.CALVADOS_frontend` — the CALVADOS2 model.

Both objects inherit from a common base class
(:class:`~finches.frontend.frontend_base.FinchesFrontend`) and therefore expose an
**identical interface**: code written against one frontend will work against the
other simply by swapping the object you instantiate. The two differ only in the
underlying coarse-grained forcefield and in a small number of model-specific
capabilities (most notably, only Mpipi supports protein–RNA interactions).

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
   * - Protein–RNA interactions (poly-U)
     - **Yes** — use ``'U'`` in a sequence
     - No — a sequence containing ``'U'`` raises ``ValueError``
   * - Solution conditions
     - ``salt``, ``dielectric``
     - ``salt``, ``pH``, ``temperature``

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

    seq1 = "MESNQSNNGGSGNAALNRGGRYVPPHLRGG"
    seq2 = "LEGMSGDMRSGGGYRGRGGRGNGQRFGGRD"

    # a single mean-field interaction parameter (negative = attractive)
    eps = mf.epsilon(seq1, seq2)

    # a publication-ready interaction map (intermap)
    fig, im, ax_main, ax_top, ax_right, ax_cbar = mf.interaction_figure(seq1, seq2)

    # a homotypic phase diagram
    data, fig, ax = mf.plot_phase_diagram(seq1)

    # Mpipi only: per-residue RNA-binding profile (vs poly-U)
    positions, rna_binding = mf.protein_nucleic_vector(seq1)

Solution conditions are set at construction time, e.g.
``Mpipi_frontend(salt=0.05, dielectric=80.0)`` or
``CALVADOS_frontend(salt=0.15, pH=5.5, temperature=310)``.


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
* :meth:`~finches.frontend.frontend_base.FinchesFrontend.protein_nucleic_vector`
  / :meth:`~finches.frontend.frontend_base.FinchesFrontend.plot_protein_nucleic_vector`
  — RNA-binding propensity along a protein (**Mpipi only**).
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

``Mpipi_frontend`` uses the Mpipi-GG forcefield and supports both protein–protein
and protein–RNA (poly-U) interactions. RNA is represented by ``'U'`` characters;
when a sequence contains ``'U'`` disorder prediction is automatically disabled for
that sequence (metapredict cannot analyse RNA).

.. autoclass:: finches.frontend.mpipi_frontend.Mpipi_frontend
   :members:
   :inherited-members:
   :show-inheritance:


.. _calvados-frontend-reference:

CALVADOS_frontend
-----------------

``CALVADOS_frontend`` uses the CALVADOS2 forcefield for protein–protein
interactions and additionally lets you set the solution ``pH`` and
``temperature``. CALVADOS2 has no RNA parameters, so any sequence containing
``'U'`` raises a ``ValueError`` and the RNA-specific methods are unavailable.

.. autoclass:: finches.frontend.calvados_frontend.CALVADOS_frontend
   :members:
   :inherited-members:
   :show-inheritance:


Shared base class
-----------------

Both frontends inherit from ``FinchesFrontend``. It is documented here for
completeness and should **not** be instantiated directly — always use one of the
derived classes above.

.. autoclass:: finches.frontend.frontend_base.FinchesFrontend
   :members:
   :show-inheritance:
