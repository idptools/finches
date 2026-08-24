# Changelog

### Version 0.1.5 (in development; 2026-08-24)
This version covers two rounds of work: a code-quality, performance, and documentation pass over the core and frontend modules, and a new module (`structure_surface.py`) for mapping IDR interactions onto folded-domain surfaces. The performance changes and refactors were all verified to produce output **identical** to the previous implementations; the items under "Bug fixes" intentionally change behaviour because they correct previously-incorrect behaviour.

**New: structure-aware surface interaction mapping (ACTIVE DEVELOPMENT - do not rely on this yet)**

> ⚠️ **`finches/utils/structure_surface.py` and its tests are in active development.** The API, the default parameters, and the numbers this code produces are all still moving, and none of it should be treated as stable or as validated science yet. It is included in this version so the work is versioned and reviewable, not because it is ready to be used in anger. In particular the polymer-reach parameters (`reach_b`, `reach_nu`), the new `contact_radius`, and the surface-classification/occlusion thresholds are **uncalibrated** - they are physically-motivated defaults, not fitted values, and `reach_b` in particular sets the footprint radius directly and has not yet had a sensitivity sweep. Expect the defaults, and therefore the output, to change.

* New `StructureSurface` class. It loads a PDB/mmCIF structure, decomposes each chain into folded domains and IDRs, classifies folded-domain surface residues by SASA, and builds a contiguous-surface net in which distances between surface residues are geodesic (over-the-surface, via Dijkstra) rather than straight-line, with occlusion pruning so that edges cannot cut through the protein core.
* `surface_vs_idr_matrix()` is the structural analogue of a FINCHES IDR:IDR intermap: every surface residue is scored against every sliding window of an IDR. `rows='all'` returns one row per residue in the structure so that buried residues can be greyed out rather than dropped.
* Interactions are modelled as a **contact footprint** rather than a point contact. The window's centre residue is treated as pinned over the row's residue; the residue `k` positions along the chain is then typically `R(k) = reach_b * k**reach_nu` Angstrom away. Because `R(k)` is a *mean* end-to-end distance rather than a hard limit, contact weight falls off as `exp(-3 d**2 / (2 R**2))` over geodesic distance `d` instead of switching off at a cutoff, and the reach is floored by a single residue's contact shell (`R_eff = sqrt(contact_radius**2 + R**2)`, default `contact_radius = 6 A`). Both details matter: a top-hat mask makes a residue flip in and out of the footprint as the pinned point shifts by an Angstrom, which puts steps into the map that the structure does not have, and a zero floor makes the window centre a point contact carrying ~20% of every cell on its own. With the current defaults, surface residues that touch each other disagree roughly half as much as distant ones, while chemically identical residues in different environments still separate.
* **Values are on the same scale as an IDR:IDR intermap.** Each cell is a sum over the IDR window of a mean over the surface each IDR residue can reach, which is the same form as a FINCHES intermap cell. With the weighting terms off, a patch-only cell reduces exactly to `InteractionMatrixConstructor.calculate_epsilon_value(window_sequence, patch_sequence)` (verified to machine precision), which is what makes surface:IDR and IDR:IDR numbers comparable.
* FINCHES charge and aliphatic weighting are applied on both sides, using the contiguous-surface patch as the structural equivalent of the +/-1 sequence window.
* Interactions can be computed in trans or in cis. In cis, `idr_tether()` resolves the anchor residue and which terminus is attached straight from the decomposition, and each IDR position is attenuated by how far it has to stray from the anchor to reach the row's residue, on the same Gaussian falloff. Cells therefore fade out with distance from the tether rather than being cut off at a boundary, and the trans result is recovered wherever the whole domain sits well inside the tether's reach.
* Helper methods to write PDB files with per-residue values in the B-factor column (solvent accessibility, surface patches, IDR interaction scores, reachability from a given anchor) for visualisation in PyMOL/ChimeraX.
* ~35 tests covering the decomposition invariants, surface net and occlusion pruning, the reach model, the units acceptance above, spatial smoothness, and the cis gating.

**Analytical Flory-Huggins (`finches.analytical_fh`)**
* `spinodal()`, `GL_binodal()`, `binodal()` and `analytic_binodal()` all included chi values exactly equal to the critical chi when passed an array. At chi_c the two binodal branches meet at phi_c and the closed forms evaluate to NaN, so an array spanning chi_c returned NaNs in an otherwise-valid result. The array paths now filter on a strict `chi > chi_c`, and the "no LLPS" guard correctly rejects `max(chi) <= chi_c`, which makes the array path consistent with the scalar path.
* `binodal()` now raises a `ValueError` for chain lengths `n < 1` instead of silently returning nonsense.
* Rewrote the subpackage README and added detailed numpy-style docstrings throughout `backend.py` and `floryhuggins.py`, including which mode to use when and the failure modes of each (the Ginzburg-Landau expansion is only meaningful very close to chi_c; the self-consistent iteration can overflow to NaN for large L and large chi).
* Extended the test suite, including that the array path never returns NaN and that it raises at chi_c exactly as the scalar path does.

**Performance**
* Pairwise interaction matrices (`InteractionMatrixConstructor.calculate_pairwise_heterotypic_matrix`) are now built with a precomputed NumPy lookup table rather than per-element dictionary lookups (~8x faster for that step; transparently falls back to the dictionary path for non-standard residue codes).
* Vectorized `get_charge_weighted_mask()` and `get_aliphatic_groups()` in `parsing_aminoacid_sequences.py`, removing per-residue Python loops. Both were verified bit-identical to the previous implementations (exhaustively for the aliphatic clustering).
* Net effect: `epsilon()` is ~2.3x faster, which compounds across DMS, null-shuffle, and the per-residue / sliding-window analyses.
* Fixed an initialization inefficiency in `Mpipi_model` where each parameter pickle was loaded 5 times (once per filename in the existence-check loop) instead of once.
* Removed `build_ref_GS_AFRC_distances()` and its module-scope invocation from `finches/data/reference_sequence_info.py`. This built a 2000-entry dictionary of AFRC end-to-end distances at import time, and nothing in the codebase read it. Worth roughly 0.28s off import (note this is *not* the dominant cost - importing `finches` at all pulls in metapredict and pytorch_lightning, which is a few seconds regardless).

**Bug fixes**
* `CALVADOS_frontend.protein_nucleic_vector()` was missing its `self` argument.
* `get_sequence_epsilon_vectors()` (`epsilon_stateless.py`) and `calculate_weighted_pairwise_matrix()` (`epsilon_calculation.py`) used `x or default`, which silently discarded an explicitly-passed `null_interaction_baseline` / `charge_prefactor` of `0`; both now use explicit `is None` checks.
* The non-Cython fallback in `calculate_sliding_epsilon()` returned the seq1/seq2 index arrays in swapped order relative to the (default) Cython path.
* The `zero_folded` logic in `interaction_figure()` had an off-by-one (it could wrap to row -1 and skipped the final row); it is now vectorized row/column masking.
* The `disorder_1` / `disorder_2` arguments to the base `interaction_figure()` were silently ignored; they are now passed through to `intermolecular_idr_matrix()`.
* `CALVADOS_frontend.interaction_figure()` did not expose the `linewidth` argument that the base implementation accepts, so domain-boundary and marker line widths could not be set from the CALVADOS frontend; it is now in the signature, documented, and passed through.
* `folded_domain_utils.py`: fixed an undefined-variable (`override_mapping`) `NameError` in the residue-override path, a non-interpolating f-string, and a misplaced `weight=` keyword that was passed to `dict()` instead of `nx.all_pairs_dijkstra_path()`.
* `InteractionMatrixConstructor.__init__()` silently accepted forcefields whose `null_interaction_baseline` or `charge_prefactor` is uncalibrated (stored as `np.nan`, as is the case for CALVADOS1), which then propagated NaN into every downstream epsilon value. It now raises a `ValueError` naming the forcefield version and the missing constant.
* `CALVADOS_model.__init__()` validated the `version` argument *after* using it to index the residue parameter dictionary, so an unknown version surfaced as a bare `KeyError` rather than the intended message. The guard now runs first.
* `finches/tests/test_data/update_test_data.py` imported `get_attractive_repulsive_matrixes` from `epsilon_stateless`, which does not exist (the function is `get_attractive_repulsive_matrices`).
* `.gitignore` contained `*c` rather than `*.c`, which silently excluded any path ending in the letter "c" - including `docs/_static/`.

**Documentation**
* Rebuilt `docs/api.rst` into a detailed reference for `Mpipi_frontend` and `CALVADOS_frontend`, including a feature-comparison table, a quickstart, and a categorized method overview.
* Corrected and standardized the numpy-style docstrings across the frontend, forcefield, analytical-FH, and domain-decomposition modules, and fixed their reStructuredText so they render cleanly under Sphinx (the API page now builds with no autodoc warnings).
* Added the FINCHES logo (`docs/media/finches_logo_v1.png`) to the documentation sidebar via `html_logo`.
* Wrote the previously-stubbed `docs/phase_diagrams.rst`, `docs/phase_diagram_caveats.rst` and `docs/intermap_caveats.rst`, and added them to the toctree in `docs/index.rst`.
* Fixed the worked examples in the docs: `docs/dms.rst` had inverted PRGn colours and an incorrect claim about alphabetical row ordering, and `docs/idr_fd.rst` referenced an undefined `mf` and was missing its numpy import. Rewrote the truncated `docs/examples.rst`.

**Repository layout**
* Renamed `demo/` to `examples/` and added `examples/folded_domains/`, which holds worked examples for the folded-domain and surface-mapping functionality (p53, NPM1, and a GCN4 IDR vs ADBD1 surface notebook for `structure_surface.py`).
* **Breaking:** removed `finches/sequence_tools.py` (and its tests). Nothing in the codebase imported it. If you were using it directly you will need to pin 0.1.4.

**Code quality**
* Ran `ruff format` on all touched modules and resolved the outstanding `ruff check` lint (unused imports/variables, duplicate imports, ambiguous names, `type(...) ==` comparisons, etc.).


### Version 0.1.4 (2026-01-28)
* Fixed bug in CALVADOS initialization (Stephen also independently fixed - thanks, Stephen!)
* Added dms() and plot_dms() functions into the `frontend_base.py` code for frontend objects
* Added additional tests for frontened objects.
* Added docs for dms() and plot_dms() functionality.
* Added nice plot formatting as a decorator into `frontend_base.py`

### Version 0.1.4 (beta; January 2026)
* Major update to internal code, including:
	* Forcefield module now has a base `ForcefieldModel` class that derived forcefield models inherit from.
	* `Mpipi_model` implemented in `mpipi.py` inherits from ForcefieldModel
	* `CALVADOS_model` implemented in `calvados.py` has been completely rewritten, also now inherits from ForcefieldModel and has had a major improvement in performance. Moved away from Pandas for parsing data to dictionaries as per Mpipi, new data file with CALVADOS parameters `calvados_residues_dict.pickle` replaces odl file (`calvados_residues.pickle`).
	* Removed a lot (although not all) redundant code
	* Reworked `epsilon_calculation.py`, `epsilon_stateless.py`, `sequence_tools.py` and `parsing_aminoacid_sequences.py` to improve performance, readability, and documentation.
	* Added >200 tests for functions in `epsilon_calculation.py`, `epsilon_stateless.py`, `sequence_tools.py`, and `parsing_aminoacid_sequences.py`, as well as created the `generate_data_test_epsilon_calculation.py`, which generated input comparison ground truth data for tests.


### Version 0.1.3 (beta; July 2025)
* Version 0.1.3 includes a number of minor bug fixes as well as updates to readme   post publication.
* Documentation added at [https://finches.readthedocs.io/en/stable/](https://finches.readthedocs.io/en/stable/)

### Version 0.1.2 (beta; Dec 2024)
* Version 0.1.2 brings a number of new features that are fully implemented but limited in their usage capacity. This will be addressed over the coming weeks (i.e., by the end of 2024).
* A new approach for automating IDR-associated domain decomposition. This domain decomposition approach was developed by Alex Keeley. 
* **InterLogos**: In 0.1.2, we're pleased to introduce InterLogos. InterLogos are akin to sequence logos in that they provide a way to visually identify residues and regions that are predicted to drive attractive intermolecular interactions. InterLogos were developed by Nick Razo. InterLogos will be introduced into finches-online in the near future.
* A newly revised method for calculating IDR:folded domain surface interactions that give residue-specific information. This approach was developed by Stephen Plassmeyer and Ryan Emenecker.
* Various other small improvements.


### Version 0.1.1 (beta)
* Added `no_disorder` flag to frontend objects, which means we generate intermaps without the disorder profiles. 
* Added `per_residue_repulsive_vector()` function to frontend objects, enabling the repulsive vector to be returned.
* Added mpipi and calvados fingerprint sequence

### Version 0.1 (beta)
* Initial public release

### Pre release
* **May 9th 2024** - fixed a stupid off-by-one bug in the indexing from the sliding window epsilon functions - basically invisible unless you're looking at matrices of ~10 residues or smaller, but even so...
* **May 5th 2024** - major update and breaking changes; **PLEASE READ**. We realized that the CALVADOS integral for calculating interaction parameters was integrating over nanometers instead of over angstroms, meaning all the CALAVDOS residue-specific interaction parameters were (consistently) smaller than they should be by a d(distance)-factor of 10. This changes nothing about any scientific insights that have been gleaned but does change the absolute numerical values of CALVADOS-derived epsilon analysis. This has now been fixed, but in the process, we used this as an opportunity to update a range of additional things, listed below:
	* As mentioned, CALVADOS epsilon parameters are now ~10x what they were before, which is more in keeping with epsilon values. This will necessitate any code that analyses or visualizes CALVADOS-associated epsilon values to update thresholds, etc., but the actual rank order and relative epsilon values should all remain approximately the same (we had to make some very small changes to the baseline correction).
	* We shifted to ensure module names are all lowercase (e.g., `finches.forcefields.mPiPi` -> `finches.forcefields.mpipi` to be consistent with Python best practices
	* We shifted `Interaction_Matrix_Constructor` to `InteractionMatrixConstructor` to again be consistent with Python best practices (camel-case for Class names).
	* We unified Mpipi type-setting as "Mpipi", instead of "mPiPi", which had been used previously. This has changed the names of classes, parameter files, etc., etc., but everything within finches *should* now be internally consistent. However, existing code may be importing now-deprecated parameter names or module names. This should be an easy fix in your local code.
	* We removed the `legacy_mpipi_frontend.py` module from `finches.frontend` - this was where the original non-inheritance-based frontend had been implemented, but we've now moved to an inheritance-based approach for frontends, which has worked without issue, so we're removing legacy entirely. 
	* Added `epsilon_vectors()` to the frontend base to enable direct access to the attractive and repulsive vectors (note this does not involve any sliding window smoothing). 
	* We re-factored the stand-alone stateless epsilon-associated functions previously found in `epsilon_calculations.py` into a new function module called `epsilon_stateless.py`. This avoids a whole host of circular dependencies and means various functions can access these broadly-applicable epsilon-associated functions WITHOUT needing to also import the code associated with the Interaction_Matrix_Constructor
	* We removed the automatic code for calculating charge_prefactor - the approach used originally was revised to a new approach, which is codified in a Jupyter notebook that will be bundled with FINCHES, but this can't be easily automated as it requires some decisions to be made, so we've removed the false sense of being able to automate this.
	* We simplified the import structure so that for most types of analysis a "standard user" might want, you can get there now in a few easy lines:

			from finches import Mpipi_frontend, CALAVDOS_frontend
			mf = Mpipi_frontend()
			cf = CALAVDOS_frontend()
	* Lots of updates to the docstrings to re-write documentation 


* April 8th 2024 - revamped `finches.frontend` module to include CALVADOS and Mpipi frontend classes that can be used for simple access to identical functionality from Mpipi or CALVADOS (or any future model). 
