.. finches documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FINCHES documentation
=========================================================

What is FINCHES?
-------------------
FINCHES is a Python package lets you predict interactions driven by intrinsically disordered regions (IDRs) in proteins. It uses a bottom-up approach to predict intermolecular interactions based on the chemical specificity of the amino acid sequence, using the chemical physics extracted from coarse-grained force fields.

FINCHES lets you do a few things:

1. Generated intermolecular interaction maps (intermap) for a pair of disordered sequences. Intermaps are a 2D representation of the predicted intermolecular interactions between two sequences, where the x-axis is one sequence and the y-axis is the other. The color of each pixel in the Intermap indicates the predicted interaction strength between the two residues at that position.

2. Generated intermap for a disordered sequence with the surface of a folded domain. In this situation, we extract out the surface residues of a given folded domain then generate an intermap between the surface residues and the IDR residues.

3. Calculate a mean-field interaction parameters (epsilon) that quantifies the average IDR interaction strength with another IDR or a folded domain. 

4. Predict the homotypic phase diagram for a disordered sequence. This is most useful when comparing how changes in a sequence are expected to alter phase behavior (as opposed to predicting the absolute phase behavior of a sequence). 

For all of these analyses, there are many caveats that should be considered. These are discussed in detail in the relevant sections of the documentation. 

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   getting_started
   background   
   epsilon
   idr_idr
   idr_fd
   phase_diagrams
   general_caveats
   api
   extended_methods
   acknowledgements



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
