Extended Methods
==================



Implementation
------------------
FINCHES is fully open source and hosted at https://github.com/idptools/finches. It is implemented in Python (https://www.python.org/) (version 3.7 or higher) with Cython implementations for a subset of performance-sensitive algorithms (https://cython.org/). FINCHES also makes extensive use of NumPy, SciPy, and Matplotlib. Version control is provided via git (https://git-scm.com/). Bugs and features can be reported/requested at https://github.com/idptools/finches/. Additional forcefields can be implemented in the finches.forcefields module.

The finches-online webserver (http://finches-online.com/) is built using Flask (https://flask.palletsprojects.com/) and uses nginx (https://nginx.org/en/) and gunicorn (https://gunicorn.org/) for backend infrastructure. It provides front-facing services for intermap construction and phase diagram prediction.


Calculating Epsilon
------------------------
The calculation of the overal mean-field :math:`\epsilon` value (:math:`\epsilon^{MF}`) is done in several steps.


Calculating Intermaps
------------------------
Intermaps describe the local pairwise interaction between subregions of an IDR and partner (either another IDR or the surface residues on a folded domain). While the overal mean-field :math:`\epsilon` value (:math:`\epsilon^{MF}`) describes the overall average attractive/repulsive interaction between an IDR and a partner, intermaps deconvolve this average value into local regions. This can be especially useful for identify which subregions within an IDR contribute attractive and repulsive interactions. 

Intermaps are calculated using the :math:`M^{raw}` matrix in a manner that is analogous to the :math:`\epsilon^{MF}`. Specifically, once the :math:`M_{raw}` matrix has been determined for the two sequences of interest, we systematically extract subsquares (of dimensions defined by some window size) from inside the :math:`M^{raw}` matrix. Each subsquare is then treated as a new but “local” :math:`M_{raw}` matrix (:math:`M^{raw-local}` matrix), and we calculate a local :math:`\epsilon^{MF}` for that subsquare. :math:`M^{raw-local}` matrices are uniformly sampled across :math:`M^{raw}` with a step size of 1. Because we build the :math:`M^{raw-local}` matrices from the original :math:`M^{raw}`, local interaction effects are captured without window boundary effects. The window size used defines the number of MRaw-local matrices extracted (which will be (sequence 1 length - window size + 1) :math:`\times` (sequence two length - window size + 1)). Because :math:`M^{raw-local}` is guaranteed to be square, the intermap obtained if two sequence orders are flipped are simply rotations of one another, such that for intermap construction, the order of the sequences does not affect the resulting map.


Sensitivity analysis
------------------------
To assess model sensitivity, we calculated εMF values for several sequences while varying the underlying forcefield parameters, allowing us to calculate the dependencies of the εMF on each individual inter-residue pair and back-calculate sensitivity in terms of the absolute % change in εMF compared to the % change applied to the underlying parameter. These analyses are shown in **Fig. S16**. In all cases examined, a change of 1% in any of the underlying forcefield parameters has a < 1% change in the resulting predictions, but there is a measurable change (typically up to ~0.5% for the largest effects). This suggests that the conclusions we arrive at do depend on the underlying forcefield parameters, but there is reasonable local robustness that if we made seemingly small updates to the parameters, our conclusions would remain qualitatively similar. We also found that – as expected – strongest dependence of a given sequence was scaled by the fraction of the amino acids present in that sequence.

This is an inline equation: :math:`E = mc^2`


Calibration of Charge Prefactor
-----------------------------------
We implemented a simple charge correction that downweighs repulsion from clusters of like-charged residues to capture local charge effects. We sought to implement the simplest charge correction possible to capture local charge effects. This goal was motivated by the fact that more complex corrections would require more parameters and decisions, and – at this stage – our goal is to implement an approach that uses the native forcefields as closely as possible. 

The motivation for such a charge correction is two-fold. Firstly, recent work has shown how the local electrochemical environment can modulate pKa values, such that tracts of acidic residues may have upshifted pKas (and tracts of basic residues may be downshifted pKas). 

The local charge correction is implemented when calculating the initial intermolecular ϵ matrix (i.e., without any sliding window analysis). If two residues in opposing chains are both charged, we excise the i+1 and i-i residues around the two residues in question and concatenate these two subsequences together to generate a fragment with a maximum length of six amino acids (minimum of four, in the case of two end residues). For that fragment, we calculate the fraction of charged residues (FCR), which lies between 0 and 1, and the net charge per residue (NCPR), which lies between -1 and +1. The weighting factor is then computed as |NCPR/FCR|. This means that for fragments of all one type of charge, the weighting factor is 1 (1/1), whereas for net neutral fragments, the weighting factor is 0 (0/1). Intermediate values are then tuned based on the fragment’s FCR and NCRP values. The weighting factor is applied to each instance where pairs of residues in opposing chains are both charged. Finally, all those weighting factors are scaled in a forcefield-specific manner via a single scalar we refer to as the charge prefactor. The charge prefactor is the only free parameter in our charge correction and allows different force fields to implement differing levels of local charge effects. The charge prefactor must lie between 0 (no local charge effects) and 1 (weighting factors used directly).

We calibrated the Mpipi-GG charge prefactor at 0.2 based on tuning the self-interaction propensity of the Das-Pappu sequences (Fig. S17) (12). Briefly, based on single-chain compaction results alongside work from Lin and Chan, we anticipate sequences with large blocks of oppositely charged residues (i.e., high-kappa sequences) to have a negative (attractive) ϵ in the zero-salt limit (13). A charge prefactor of 0.2 defines the boundary between attractive and repulsive ϵ values for these sequences in a reasonable location, given previous theoretical predictions and simulation work (Fig. S18A) (13–15). For CALVADOS2, the charge prefactor was calibrated to match the (magnitude-corrected) trends described by Mpipi-GG, giving a charge prefactor of 0.7 (Fig. S18B, C). This calibration is relatively empirical and qualitative. While a more robust calibration is certainly possible, we suggest attempting to fit a more precise dependency here would over-extend the reasonable predictive power of the underlying force field, especially given the layers of simplifying assumptions being made. 
