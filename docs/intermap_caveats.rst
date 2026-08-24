Intermap caveats and considerations
====================================

The caveats below are specific to intermolecular interaction maps, whether IDR:IDR or IDR:folded domain. They sit on top of the :doc:`general_caveats`, which apply to every FINCHES calculation, and which include the important point that an intermap describes the potential for interaction rather than the interaction you would actually observe.

Intermaps depend on the window size
.....................................
Every value in an intermap is the epsilon between a window of one sequence and a window of the other, with a default window of 31 residues. The window size is not a cosmetic choice. Larger windows average over more sequence context and produce smoother maps with broader, weaker-looking features; smaller windows are noisier but resolve shorter motifs. A feature that appears at one window size may soften or disappear at another, so if you are making a claim about a specific region it is worth confirming the feature survives a reasonable range of window sizes.

Window size also sets the smallest region you can say anything about. There is no sense in which a 31-residue window can localize an interaction to a single residue, and the per-residue axes of the plot should not be over-read.

The edges of every intermap are missing
.............................................
Because each value requires a full window centred on a position, roughly half a window at each terminus of each sequence cannot be evaluated and is trimmed. With the default window this removes about 15 residues from each end of each axis. Terminal regions are therefore invisible to this analysis, which matters more often than people expect - N- and C-terminal tails are frequently where interesting things happen.

Colour scales are not comparable between force fields
........................................................
The default colour limits differ between the frontends: the Mpipi frontend defaults to a range of -3 to +3, while the CALVADOS frontend defaults to -7.5 to +7.5. This reflects the different numerical ranges the two force fields produce, but it means that two intermaps drawn with default settings are not visually comparable, and neither are the underlying values. As noted in the general caveats, pick one force field for a given analysis and stay with it. If you do need to place two maps side by side, set ``vmin`` and ``vmax`` explicitly and identically, and be clear about what is being compared.

In the default colour map, purple indicates attractive interactions and green indicates repulsive ones.

Disorder prediction is doing work in the background
........................................................
By default the frontends run metapredict over the input sequences and use the result to mask folded regions out of the figure. This is a prediction, and it inherits whatever errors that prediction makes. A region wrongly called folded will be zeroed out of the map, and a region wrongly called disordered will be included and interpreted with a model that assumes a disordered ensemble. If a map looks surprising in a particular region, checking the disorder profile is a sensible first diagnostic. The masking behaviour can be turned off, in which case the entire sequence is treated as disordered.

Consider a shuffled null
.....................................
An intermap between two sequences of a given amino acid composition will show structure simply because of that composition, before any sequence-specific effect is considered. The ``null_shuffle`` argument addresses this: passing an integer runs that many shuffled controls and subtracts the mean shuffled matrix from the raw matrix, leaving the signal attributable to the particular arrangement of residues rather than to composition alone. A value of 100 is a reasonable default. If your interpretation depends on a feature being sequence-specific rather than composition-driven, this is the check to run.

Interactions are pairwise and context-free
.............................................
Each cell of the map is computed between two fragments in isolation. There is no competition between regions, no accounting for the fact that a region can only be engaged once, and no long-range correlation along either chain. A promiscuous region will therefore light up against many partners simultaneously, which is a statement about its chemistry and not a prediction that all of those interactions occur together.

Additional considerations for IDR:folded domain maps
........................................................
Surface intermaps carry everything above plus a dependence on the structure you supply. Surface residues are identified from solvent accessibility calculated on a single input structure, so the result reflects that particular conformation - a crystal structure, one AlphaFold model, or one frame of a simulation. Alternative conformations, missing loops, bound cofactors and crystallization artefacts all propagate into the map. The folded domain is also treated as rigid, and the analysis says nothing about conformational change on binding.
