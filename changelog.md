# Changelog

### Version 0.1 (beta)
* Initial public release

### Pre release
* **May 9th 2024** - fixed a stupid off-by-one bug in the indexing from the sliding window epsilon functions - basically invisible unless you're looking at matrices of ~10 residues or smaller, but even so...
* **May 5th 2024** - major update and breaking changes; **PLEASE READ**. We realized that the CALVADOS integral for calculating interaction parameters was integrating over nanometers instead of over angstroms, meaning all the CALAVDOS residue-specific interaction parameters were (consistently) smaller than they should be by a d(distance)-factor of 10. This changes nothing about any scientific insights that have been gleaned but does change the absolute numerical values of CALVADOS-derived epsilon analysis. This has now been fixed, but in the process, we used this as an opportunity to update a range of additional things, listed below:
	* As mentioned - CALVADOS epsilon parameters are now ~10x what they were before, which is more in-keeping with epsilon values. This will necessitate to any code that analyses or visualizes CALVADOS-associated epsilon values to update thresholds etc, but the actual rank order and relative epsilon values should all remain approximately the same (we had to make some very small changes to the baseline correction).
	* We shifted to ensure module names are all lowercase (e.g., `finches.forcefields.mPiPi` -> `finches.forcefields.mpipi` to be consistent with Python best practices
	* We shifted `Interaction_Matrix_Constructor` to `InteractionMatrixConstructor` to again be consistent with Python best practices (camel-case for Class names).
	* We unified Mpipi type-setting as "Mpipi", instead of "mPiPi", which had been used previously. This has changed the names of classes, parameter files, etc etc, but everything within finches *should* now be internally consistent. However, existing code may be importing now deprecated parameter names or module names. This should be an easy fix in your local code.
	* We removed the `legacy_mpipi_frontend.py` module from `finches.frontend` - this was where the original non-inheritance-based frontend had been implemented, but we've now moved to an inheritance-based approach for frontends which has worked without issue, so we're removing legacy entirely. 
	* Added `epsilon_vectors()` to the frontend base to enable direct access to the attractive and repulsive vectors (note this does not involve any sliding window smoothing). 
	* We re-factored the stand-alone stateless epsilon-associated functions previously found in `epsilon_calculations.py` into a new function module called `epsilon_stateless.py`. This avoids a whole host of circular dependency and means various functions can access these broadly-applicable epsilon-associated functions WITHOUT needing to also import the code associated with the Interaction_Matrix_Constructor
	* We removed the automatic code for calculate charge_prefactor - the approach used originally was revised to a new approach which is codified in a jupyter notebook which will be bundled with finches, but this can't be easily automated as it requires some decisions to be made, so we've removed the false sense of being able to automated this.
	* We simplified the import structure so that for most types of analysis a "standard user" might want, you can get there now in a few easy lines:

			from finches import Mpipi_frontend, CALAVDOS_frontend
			mf = Mpipi_frontend()
			cf = CALAVDOS_frontend()
	* Lots of updates to the doc strings to re-write documentation 


* April 8th 2024 - revamped `finches.frontend` module to include CALVADOS and Mpipi frontend classes that can be used for simple access to identical functionality from Mpipi or CALVADOS (or any future model). 
