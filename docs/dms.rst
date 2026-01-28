Deep Mutational Scanning (DMS)
================================

What is DMS in FINCHES?
------------------------

FINCHES provides deep mutational scanning (DMS) functionality that allows you to systematically evaluate how every possible single-point mutation in a protein sequence affects its homotypic (self-interaction) epsilon value. This is useful for:

1. Identifying which residues are most critical for driving self-interactions (e.g., phase separation propensity)
2. Predicting which mutations might enhance or diminish self-interaction strength
3. Understanding the sequence-level determinants of homotypic interactions

The DMS analysis generates a 20 × n matrix, where n is the sequence length, and each element represents the epsilon value (or change in epsilon) for a specific amino acid substitution at a specific position.

How to perform DMS with FINCHES
---------------------------------

Basic DMS calculation
~~~~~~~~~~~~~~~~~~~~~~

The ``dms()`` function calculates homotypic epsilon values for all possible single-point mutants:

.. code-block:: python

    from finches import Mpipi_frontend

    # initialize the frontend
    mf = Mpipi_frontend()

    # example sequence
    seq = 'MSKGEELFTGVVPILVELD'

    # perform DMS scan
    matrix, amino_acids, positions = mf.dms(seq)

    # matrix is 20 x 19 (20 amino acids x 19 positions)
    print(matrix.shape)
    >> (20, 19)

    # amino_acids lists the amino acids in row order
    print(amino_acids)
    >> ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    # positions gives 1-indexed position numbers
    print(positions)
    >> [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]


Calculating delta epsilon values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often it's more informative to look at the *change* in epsilon relative to the wild-type sequence. Use ``return_delta=True``:

.. code-block:: python

    # get delta values (mutant epsilon - wild-type epsilon)
    matrix, amino_acids, positions = mf.dms(seq, return_delta=True)

    # negative values = mutation makes sequence MORE self-attractive
    # positive values = mutation makes sequence LESS self-attractive

    # find the most stabilizing mutation
    import numpy as np
    min_idx = np.unravel_index(np.argmin(matrix), matrix.shape)
    best_aa = amino_acids[min_idx[0]]
    best_pos = positions[min_idx[1]]
    original_aa = seq[min_idx[1]]
    print(f"Most stabilizing mutation: {original_aa}{best_pos}{best_aa}")


Visualizing DMS results
-------------------------

The ``plot_dms()`` function provides a convenient way to visualize DMS results as a heatmap:

Basic heatmap
~~~~~~~~~~~~~~

.. code-block:: python

    from finches import Mpipi_frontend
    import matplotlib.pyplot as plt

    mf = Mpipi_frontend()
    seq = 'MSKGEELFTGVVPILVELDGDVNGH'

    # generate DMS heatmap
    fig, ax, im, dms_data = mf.plot_dms(seq)
    plt.show()

By default, ``plot_dms()`` uses ``return_delta=True``, so the heatmap shows the change in epsilon upon mutation. Green regions (negative values in the default PRGn colormap) indicate mutations that increase self-attraction, while purple regions (positive values) indicate mutations that decrease self-attraction.

Customizing the heatmap
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # customize color scale and appearance
    fig, ax, im, dms_data = mf.plot_dms(
        seq,
        vmin=-2,                    # minimum value for color scale
        vmax=2,                     # maximum value for color scale
        cmap='coolwarm',            # alternative colormap
        figsize=(10, 5),            # custom figure size
        tic_frequency=5,            # position label frequency
        show_wt_marker=True,        # mark wild-type amino acids
        wt_marker_color='red',      # color for WT markers
        wt_marker_size=5,           # size of WT markers
    )
    plt.show()

Showing the sequence
~~~~~~~~~~~~~~~~~~~~~

You can display the amino acid sequence directly below the heatmap:

.. code-block:: python

    fig, ax, im, dms_data = mf.plot_dms(
        seq,
        show_sequence=True,         # display sequence below heatmap
        sequence_fontsize=6,        # font size for sequence letters
    )
    plt.show()

Saving the figure
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # save directly to file
    mf.plot_dms(seq, fname='dms_heatmap.png')

    # or use matplotlib
    fig, ax, im, dms_data = mf.plot_dms(seq)
    fig.savefig('dms_heatmap.pdf', bbox_inches='tight')


Interpreting DMS results
--------------------------

Understanding the heatmap
~~~~~~~~~~~~~~~~~~~~~~~~~~

In a delta-epsilon DMS heatmap:

- **Rows** represent the 20 standard amino acids (in alphabetical order by single-letter code)
- **Columns** represent positions in the sequence
- **Color** represents the change in homotypic epsilon upon mutation
- **Wild-type markers** (black circles by default) indicate the native amino acid at each position

Key patterns to look for:

1. **Vertical stripes**: Positions where many mutations have similar effects, suggesting the position is either critical (strong effects) or tolerant (weak effects) regardless of substitution
2. **Horizontal stripes**: Amino acids that consistently increase or decrease self-interaction when introduced anywhere in the sequence
3. **Hotspots**: Individual positions where specific substitutions have unusually large effects

Biological interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Negative delta values** (more attractive): The mutation increases homotypic interaction strength, potentially enhancing phase separation propensity
- **Positive delta values** (less attractive): The mutation decreases homotypic interaction strength, potentially reducing phase separation propensity
- **Wild-type values near zero**: Mutations at these positions have minimal effect on self-interaction

Caveats and considerations
----------------------------

1. **Computational cost**: DMS scans require calculating epsilon for 20 × n mutant sequences. For long sequences, this can be time-consuming. Use ``show_progress=True`` (default) to monitor progress.

2. **Single mutations only**: This analysis considers only single-point mutations. Epistatic effects from combining multiple mutations are not captured.

3. **Homotypic only**: The ``dms()`` function calculates *homotypic* (self-interaction) epsilon values. Heterotypic interactions with other sequences are not evaluated.

4. **Mean-field approximation**: Like all FINCHES epsilon calculations, DMS results reflect mean-field predictions. They do not account for local structural effects or context-dependent interactions.

5. **Relative comparisons**: DMS results are most useful for comparing mutations to each other and to wild-type, rather than for predicting absolute phase behavior.
