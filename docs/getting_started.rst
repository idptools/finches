Getting Started
==================


News
--------------------------------
* **[2025-05-22: FINCHES is published]**  | We are actively working on overhauling aspects of both the codebase and the docs to fully exposure the analyses that FINCHES enables, but for now we provide a skeleton version here for conducting the types of analysis described in the paper. Over the next few months we will be adding more examples and also changing some aspects of the interface and code to make things faster and easier.

What should I use FINCHES for?
--------------------------------

We see FINCHES as being useful for a few types of roles:

* **Hypothesis generation**: The primary goal of FINCHES is to enable rapid and physically-motivated hypothesis generation for understanding sequence-to-function relationships in IDRs. Most usefully, intermaps can be used to identify regions of an IDR that are likely to be involved in intermolecular interactions, and to generate testable hypotheses about the role of specific residues in mediating these interactions. 

* **Data interpretation**: In addition to hypothesis generation, FINCHES can also be used to interpret experimental data. For example, FINCHES enables large-scale structural bioinformatics through the lense of surface chemistry, or interpreting high-throughput experiments based on FINCHES interaction analysis. 

* **Enabling the rational design of IDRs**: Finally, we have enabled the rational design of disordered regions with desired chemical specificity in the `software package GOOSE <https://goose.readthedocs.io/>`_. 

Before using FINCHES PLEASE read the relevant **Caveats and considerations** section from the documentation. This will hopefully ensure you avoid over-interpreting any prediction. We emphasize tere are many caveats to consider, and strongly encourage you to consider some of the most common considerations prior to acting on results from FINCHES,

Installation
-------------------
FINCHES can be installed using pip. To install the latest version, run the following command in your terminal::

    # create a new conda environment (recommended but not strictly required)
    conda create -n finches python=3.11 -y
    conda activate finches

    # install directly from github; this will handle all dependencies
    pip install git+https://git@github.com/idptools/finches.git

To test this has worked correctly, run the following command in your terminal::

    python -c "import finches; print(finches.__version__)"

.. note::

    The very first time you run this it will take a long time (60-90 seconds) to initialize your Python environment.
    However, after the initial time, import will be quick.


Using FINCHES:
-------------------
We recommend folks use the front-end objects for working with FINCHES. There are a lot of sometimes quite confusing backend functions which do extend the power of FINCHES further, but the frontend objects (Mpipi_frontend and CALVADOS_frontend) objects implement the core user-facing algorithms we anticipate folks will need. If specific types of analyses would be useful, we are happy to either considering incoporating them into the Frontend object base class, or providing code showing how this can be done.

All of the examples in the documentation here make use of the frontend objects. Mpipi_frontend and CALVADOS_frontend provide identical interfaces to Mpipi-GG and CALVADOS2 implementations of FINCHES.


Development:
-------------------
FINCHES is in active development with new features planned over the summer of 2025. If you have specific feature requests or identify bugs, please open an issue on the GitHub repository.


How to cite:
-------------------
If you find FINCHES useful, please cite using:

Ginell, G. M., Emenecker, R. J., Lotthammer, J. M., Keeley, A. T., Plassmeyer, S. P., Razo, N., Usher, E. T., Pelham, J. F. & Holehouse, A. S. Sequence-based prediction of intermolecular interactions driven by disordered regions. Science 388, eadq8381 (2025). DOI:10.1126/science.adq8381
  
**In addition, if you use the Mpipi version, please also cite:**

Joseph, J. A., Reinhardt, A., Aguirre, A., Chew, P. Y., Russell, K. O., Espinosa, J. R., Garaizar, A. & Collepardo-Guevara, R.
Physics-driven coarse-grained model for biomolecular phase separation with near-quantitative accuracy.
Nat Comput Sci 1, 732–743 (2021).

**If you use the CALVADOS version please also cite:**

Tesei, G. & Lindorff-Larsen, K.
Improved predictions of phase behaviour of intrinsically disordered proteins by tuning the interaction range.
Open Res. Eur. 2, 94 (2022).



