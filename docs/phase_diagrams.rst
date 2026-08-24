Predicting phase diagrams
====================================

What FINCHES predicts
.....................................
FINCHES can take a single disordered sequence and predict the homotypic phase diagram associated with it - that is, the coexistence curve (binodal) describing the concentrations of the dilute and dense phases as a function of temperature. The calculation runs in two steps. First we compute the homotypic epsilon for the sequence, exactly as described in the :doc:`epsilon` section. Second, we use that epsilon value, together with the chain length, to parameterize a Flory-Huggins free energy and solve for the phase boundaries analytically.

The key thing to understand up front is that all of the sequence chemistry enters this calculation through a single number. Once epsilon has been computed, the phase diagram depends only on that scalar and on the length of the chain. This makes the approach well-suited to asking how a change to a sequence is expected to shift phase behaviour, and poorly suited to predicting the absolute phase behaviour of any individual sequence. The :doc:`phase_diagram_caveats` section discusses this in more detail, and we strongly recommend reading it before using these predictions in anger.

Plotting a phase diagram
.....................................
The simplest entry point is ``plot_phase_diagram()``, which computes epsilon, builds the diagram, and draws it in one step.

.. code-block:: python

    from finches import Mpipi_frontend
    import matplotlib.pyplot as plt

    mf = Mpipi_frontend()

    seq = 'FYWFYWFYWFYWFYWFYWFY'

    B, fig, ax = mf.plot_phase_diagram(seq)
    plt.show()

The returned ``B`` is the underlying phase diagram data (described below), while ``fig`` and ``ax`` are the matplotlib objects, so the figure can be adjusted after the fact or saved directly with the ``filename`` keyword. The axis limits, line colour, line style, and figure dimensions are all exposed as keyword arguments, and ``xlog=True`` switches the volume fraction axis to a log scale, which is usually the more informative way to look at the dilute arm.

Getting the underlying data
.....................................
If you want the numbers rather than a figure, ``build_phase_diagram()`` returns them directly as an 8-element list.

.. code-block:: python

    B = mf.build_phase_diagram(seq)

    dilute      = B[0]   # dilute-phase volume fractions, binodal
    dense       = B[1]   # dense-phase volume fractions, binodal
    crit_phi    = B[2][0]  # critical volume fraction
    crit_T      = B[2][1]  # critical temperature
    temps       = B[3]   # temperatures matching B[0] and B[1]

    # elements 4-7 are the equivalent quantities for the spinodal
    S_dilute, S_dense, S_crit, S_temps = B[4], B[5], B[6], B[7]

Elements 0 through 3 describe the binodal and elements 4 through 7 describe the spinodal, in the same order. The critical temperature ``B[2][1]`` is usually the single most useful number here - it is the quantity that moves when a sequence is made more or less self-attractive, and it is what we generally use when ranking variants against one another.

Comparing sequences
.....................................
Because these predictions are most defensible as relative statements, the common use case is overlaying several sequences on one set of axes. ``plot_multiple_phase_diagrams()`` takes a dictionary mapping a label to a ``[sequence, colour]`` pair.

.. code-block:: python

    seq_dict = {'WT'  : ['FYWFYWFYWFYWFYWFYWFY', 'black'],
                'F1A' : ['AYWFYWFYWFYWFYWFYWFY', 'red'],
                'F1D' : ['DYWFYWFYWFYWFYWFYWFY', 'blue']}

    all_diagrams, fig, ax = mf.plot_multiple_phase_diagrams(seq_dict)
    plt.show()

The function returns a list holding the full phase diagram data for each sequence, in the order the dictionary was iterated, alongside the figure and axis objects.

Passing ``tc_ref`` with one of the dictionary keys rescales every temperature axis by the critical temperature of that reference sequence, and relabels the y-axis as :math:`T/T_c`.

.. code-block:: python

    all_diagrams, fig, ax = mf.plot_multiple_phase_diagrams(seq_dict, tc_ref='WT')

This is worth doing when you care about whether the shape of the coexistence curve changes, rather than about where it sits. Without it, differences in critical temperature dominate the plot and it becomes hard to see anything else.

How the calculation works
.....................................
The conversion from epsilon to a phase diagram happens in ``finches.epsilon_to_FHtheory.epsilon_to_phase_diagram()``, and proceeds as follows.

The homotypic epsilon is divided by the sequence length to give a per-residue interaction energy, because epsilon is an extensive quantity in the length of the chain whereas the Flory-Huggins treatment needs a site-specific energy - chain length is accounted for separately in the binodal calculation. The sign is also reversed, so that a negative (attractive) epsilon becomes a positive Flory chi.

The binodal is then solved analytically over a grid of chi values, and chi is converted to temperature using :math:`T = \Delta\epsilon/\chi`. The spinodal is computed the same way. The resulting temperature axis is therefore in arbitrary units set by the epsilon-to-chi conversion, not in Kelvin or any other physical temperature scale.

Reading the diagram
.....................................
The binodal separates a one-phase region above the curve, where the solution is homogeneous, from a two-phase region below it, where the system separates into coexisting dilute and dense phases. The left arm of the curve is the dilute-phase concentration and the right arm is the dense-phase concentration; at any temperature below the critical point, a horizontal line between the two arms is the tie line connecting the two coexisting concentrations. The critical point sits at the top of the curve.

A higher critical temperature means the sequence is predicted to phase separate more readily. When comparing variants, the direction and rough magnitude of the shift in critical temperature is the signal to pay attention to.

Caveats
.....................................
Phase diagram predictions carry a number of important limitations that are specific to this analysis, over and above the :doc:`general_caveats` that apply to everything FINCHES does. These are covered in :doc:`phase_diagram_caveats`, which we would encourage you to read.
