Examples
====================================

Worked examples are distributed as Jupyter notebooks in the ``examples/`` directory of the `FINCHES GitHub repository <https://github.com/idptools/finches>`_. Each notebook is self-contained and can be run against a local installation of FINCHES.

The notebooks are organized by the kind of question being asked:

* ``examples/overview_uses/`` - a general tour of the package, and the best place to start if you are new to FINCHES.
* ``examples/protein_matrix/`` - building and interpreting IDR:IDR interaction matrices.
* ``examples/folded_domains/`` - mapping IDR interactions onto folded domain surfaces with ``StructureSurface`` (p53, the NPM1 pentamer read from mmCIF, and the GCN4 IDR against ADBD1, which walks through the surface:IDR intermap in detail).
* ``examples/phase_diagrams/`` - converting epsilon values into Flory-Huggins phase diagrams.
* ``examples/docs_demo/`` - the notebook behind the worked examples in this documentation (epsilon, IDR:IDR intermaps, and the IDR:folded domain intermap).

The ``examples/obsolete/`` directory holds notebooks from earlier versions of FINCHES that use since-removed interfaces; they are kept for reference only and are not expected to run against the current code.

The narrative documentation in the preceding sections covers the same functionality in more detail, so the notebooks are best treated as templates to adapt rather than as tutorials to work through in order.
