Phase diagram prediction caveats and considerations
=======================================================

The caveats below are specific to phase diagram predictions. They sit on top of the :doc:`general_caveats`, which apply to every FINCHES calculation, and on top of the caveats associated with epsilon itself, since epsilon is the sole input to this analysis.

These are relative predictions, not absolute ones
.......................................................
This is the single most important thing to understand. FINCHES phase diagrams are useful for asking how phase behaviour is expected to change when a sequence changes - between a wild-type and a variant, across a mutational series, or along a designed sequence trajectory. They are not calibrated to predict the actual saturation concentration or the actual cloud point of any individual protein, and should not be quoted as such. If you take one thing from this page, take this.

The temperature axis is in arbitrary units
.......................................................
Chi is converted to temperature via :math:`T = \Delta\epsilon/\chi`, which sets the scale of the temperature axis by the magnitude of the per-residue interaction energy. The resulting axis is in arbitrary units. It is not in Kelvin, it is not in degrees Celsius, and a critical temperature of, say, 300 does not correspond to 300 K. Note that some internal variable names in the codebase unhelpfully suggest Kelvin; they do not mean it. Temperatures are meaningful when compared against other temperatures computed the same way, and not otherwise.

All sequence information is compressed into one number
.......................................................
Once the homotypic epsilon has been calculated, the only inputs to the Flory-Huggins treatment are that scalar and the chain length. Any two sequences with the same length and the same epsilon will produce identical phase diagrams, regardless of how differently their residues are arranged. Sequence patterning, the distribution of stickers along the chain, and the resulting network topology of the dense phase are all invisible at this stage, even though patterning is well known to matter a great deal for real phase behaviour. Where patterning is the question you care about, the intermap and per-residue analyses are the better tools.

Non-phase-separating sequences are not ranked meaningfully
.............................................................
A positive (net repulsive) epsilon cannot be converted into a sensible phase diagram - the chi-to-temperature conversion assumes a site-specific energy that cannot be repulsive, and a positive value produces inverted diagrams or divides by zero. FINCHES handles this by clamping any positive epsilon to a small attractive value of -0.01 before building the diagram. The practical consequence is that every sequence with a repulsive epsilon collapses onto essentially the same diagram with a very low critical temperature. You can read that as "this sequence is not predicted to phase separate", but you cannot rank two such sequences against one another, because the number that distinguished them has been discarded. If you are comparing sequences in this regime, compare their epsilon values directly instead.

Flory-Huggins is a mean-field theory
.......................................................
The underlying model assumes a homogeneous mixture of chains and solvent on a lattice, with interactions captured by a single averaged parameter. It has no notion of chain conformation, of specific contacts, of finite-size clusters, or of the network structure of a condensate. It also assumes the system is at equilibrium and that the dense phase is a simple liquid. Real condensates routinely violate several of these assumptions at once.

Homotypic, single-component systems only
.......................................................
``build_phase_diagram()`` and its plotting wrappers describe one sequence interacting with itself in a two-component system of protein and solvent. They say nothing about multi-component phase behaviour, about the effect of a binding partner or of RNA, or about which of several species will partition into a given condensate. Heterotypic epsilon values can be computed, but they are not accepted by these functions.

Solution conditions enter only through epsilon
.......................................................
Salt, pH, dielectric and temperature are set when the frontend object is constructed and affect the phase diagram only via their effect on the computed epsilon. The temperature axis of the phase diagram is not the same thing as the temperature parameter of the underlying forcefield, and varying one does not vary the other. If you want to explore how solution conditions change predicted phase behaviour, construct separate frontend objects with different conditions and compare the resulting diagrams, and note the associated caveat about comparing values computed under different parameterizations.

The spinodal is returned but is rarely what you want
.......................................................
Elements 4 through 7 of the returned data describe the spinodal, the limit of metastability. It is included for completeness and for anyone who wants to reason about nucleation barriers, but for most purposes the binodal is the curve of interest, and it is the only one drawn by the plotting functions.
