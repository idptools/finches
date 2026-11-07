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
* 45 tests covering the decomposition invariants, surface net and occlusion pruning, the reach model, the units acceptance above, spatial smoothness, and the cis gating.

**Frontend**
* Added `protein_peptide_vector()` and `plot_protein_peptide_vector()` to the frontend base class, so both `Mpipi_frontend` and `CALVADOS_frontend` have them. These mirror `protein_nucleic_vector()` / `plot_protein_nucleic_vector()` (same `fragsize`, `smoothing_window` and `poly_order` arguments, same `[positions, values]` return) but let you challenge a protein sequence with an arbitrary peptide rather than poly-U RNA.

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


**Review pass (September 2026): systematic bug sweep**

A line-by-line review of every module, with each suspected bug reproduced before it was fixed. Items marked *output-changing* alter numbers that existing code may have relied on; everything else changes error handling, documentation, or behaviour that was previously broken outright.

*Core epsilon calculation*
* `get_attractive_repulsive_matrices()` (and the Cython `matrix_scan_legacy`) dropped a matrix element that was exactly equal to the null baseline from *both* the attractive and repulsive matrices, so it contributed `-2b` rather than `x - 2b`. This made `calculate_epsilon_value()` discontinuous at the baseline and inconsistent with the summed-area-table `matrix_scan`. Such an element is now counted once. No shipped lookup table contains a value exactly equal to its baseline, so results are unchanged in practice.
* `calculate_pairwise_heterotypic_matrix()` validated sequences only on the non-default `convert_to_custom=False` path. Empty sequences, unknown residues, non-latin-1 characters and mixed protein/RNA sequences now raise a `ValueError` on every path instead of a `TypeError`, `KeyError`, `UnicodeEncodeError`, or being silently scored.
* `calculate_sliding_epsilon()` raises the same `ValueError` for an over-long window on the Cython and Python paths; `get_charge_weighted_mask()` no longer raises on an empty sequence; the benchmark script's `cython=False` path raised `UnboundLocalError`.

*Forcefields*
* **CALVADOS (output-changing for condition scans).** Salt, pH and temperature were baked into the Yukawa parameters and the histidine charge when the model was built, so setting `model.salt`, `model.pH` or `model.temperature` afterwards, which is exactly what the condition-scanning functions do, had no effect: every CALVADOS salt or pH scan returned identical epsilons and phase diagrams. The model now rebuilds those parameters whenever a condition changes, and `compute_interaction_parameter(..., salt=)` honours its `salt` argument instead of ignoring it. The lookup table at the construction conditions is bit-identical to before.
* Mpipi: the Wang-Frenkel closed form was evaluated beyond its 3-sigma cutoff, where it tends to `-eps*alpha` rather than zero. The potential is now zero past the cutoff. The integral that defines the interaction parameters already stopped at 3-sigma, so the lookup table and every epsilon are unchanged; only the raw potential profile beyond 3-sigma differs.
* `wang_frenkel()`, `coulomb()`, `compute_interaction_parameter()` and the CALVADOS pair potentials accept list input as their docstrings promised.

*Flory-Huggins and phase diagrams*
* `build_SALT_dependent_phase_diagrams()`, `build_PH_dependent_phase_diagrams()` and `build_DIELECTRIC_dependent_phase_diagrams()` now (i) regenerate the CALVADOS condition-dependent parameters described above, (ii) refuse a condition the model cannot vary (checked against `parameters.conditions`) instead of silently returning the same diagram for every value, as a pH scan on Mpipi used to, (iii) restore the model's original condition in a `finally` block, and (iv) share one implementation and signature, with `prefactor` and `null_interaction_baseline` optional on all three.
* `epsilon_to_phase_diagram()` clamps an epsilon of exactly zero as its docstring and tests describe; it previously returned all-zero temperatures for `epsilon == 0.0`.
* The accuracy claims for `analytic_binodal` in the `analytical_fh` README and docstrings were wrong. Measured against an exact equal-chemical-potential, equal-osmotic-pressure solution, the closed form is within roughly 1-8% on the dense branch for N up to ~100 but 25-50% off for N ~ 1000-5000 at 1.3-1.5 chi_c, and the dilute branch is off by orders of magnitude at large N*chi. The Newton-Raphson `binodal` converges to the exact solution wherever it does not overflow. The docs now say so.

*Frontends*
* `protein_nucleic_vector()` and `protein_peptide_vector()` returned 0-based window centres while every other frontend position vector, and the axes of their own plotting functions, are 1-based, so the trace sat one residue to the left of its labels. Positions now start at `(fragsize + 1) / 2`. *Output-changing* (positions shift by +1; values are unchanged).
* `protein_peptide_vector()` divided by the peptide length, but the underlying epsilon is already a mean over the peptide, so a 12-mer repeat of a 6-mer gave exactly half the value for identical chemistry. It now divides by the fragment length, mirroring `protein_nucleic_vector()`. *Output-changing* by the factor `fragsize / len(peptide)`.
* `interaction_figure(zero_folded=True)` and the folded-domain shading in `plot_protein_nucleic_vector()` / `plot_protein_peptide_vector()` treated metapredict's 0-based, half-open domain boundaries as 1-based, zeroing or shading one extra disordered residue on the N-terminal side of every folded domain.
* `CALVADOS_frontend.epsilon(seq1=..., seq2=...)` with keyword arguments raised `IndexError` in the RNA check; `intermolecular_idr_matrix(null_shuffle=100.0)` raised `TypeError`; `plot_attractive_logo()` crashed on a pair with no attractive contacts; `CALVADOS_frontend.protein_nucleic_vector()` raises the documented `ValueError` rather than a bare `Exception` and shares the base class's default `fragsize`. Docstrings that had the PRGn colours inverted, the DMS heatmap orientation wrong, or described a `plot_rectangles` format the code does not accept were corrected.

*Domain decomposition*
* `extract_regions()` / `convert_intermap_index_to_residue_number()` reported region starts one residue too early (a 0-based centre index used as a 1-based residue number) and converted the exclusive end of each region as if it were inclusive; the "full context" sequences could come out empty as a result. Region starts now move by +1, ends are unchanged, and the extracted sequences are exactly the window centres. *Output-changing* for `InteractingRegions` starts.
* `get_interaction_regions()` dropped the last row and column of every peak, so single-pass regions were asymmetric. `InteractingRegions.__str__` printed the length one short.

*Folded domains and structure surfaces*
* `StructureSurface`: residues missing from the coordinates were given synthetic 1-based numbers that could collide with the numbers of modeled residues. A chain numbered 10-90 with a gap at 60-62 silently replaced residues 51-53 in every lookup. Missing residues now carry the author number of the gap they fill.
* `StructureSurface`: `surface_vs_idr()` and `surface_vs_surface()` subtracted the null baseline once, while the intermap, `calculate_epsilon_value()` and everything else in FINCHES subtract it twice, so their zero did not correspond to a poly(GS)-like contact and their numbers were not on the epsilon scale. All three now share the FINCHES convention. *Output-changing*: a constant shift of `-b` per residue (+0.129 for Mpipi-GG, +0.45 for CALVADOS2).
* `StructureSurface`, `rows='all'`: the structure's own IDR residues were scored as single residues against each window, so their rows were three to five times wider than the surface rows and swamped the colour scale. They are now aggregated over their own disordered segment on the same contact-footprint model (with the footprint off this reduces exactly to an IDR:IDR intermap cell). In a cis calculation those rows are `np.nan`, since a residue with no fixed position has no distance from the anchor. *Output-changing* for IDR rows only; surface rows are bit-identical.
* `StructureSurface` and `FoldedDomain`: residue names outside the standard twenty (HID/HIE/HIP, CYX, ASH/GLH, LYN, MSE and so on) were silently turned into glycine by `StructureSurface`, and skipped by `FoldedDomain`, which misaligned the sequence against its per-residue SASA array from that residue onwards; `FoldedDomain` also counted waters and ligands as residues, with the same misalignment. Both now translate names through a shared `one_letter_code()` (standard names, then the common variants, then mdtraj's table); `StructureSurface` warns and keeps a glycine placeholder for anything else, `FoldedDomain` raises with a pointer to `residue_overide_mapping`, and `FoldedDomain` keeps protein residues only.
* `StructureSurface`: a metapredict failure used to silently treat the whole chain as folded; it now warns and says so.
* New: `StructureSurface.plot_surface_vs_idr_matrix()` and the module-level `plot_surface_intermap()` draw a surface:IDR intermap with a symmetric colour scale set from the data (99th percentile of the absolute values by default), buried residues greyed out, chain separators and residue-number axes. `footprint_order()` and `anchor_position()` are now public, and the object has a `repr`.
* `FoldedDomain.calculate_IWD()` named a distance mode that does not exist in its error message; the docstrings of `calculate_idr_surface_patch_interactions()`, `write_epsilon_vis_file()` and `extract_and_write_domains()` now describe what the code does (including that neighbour strings include the centre residue, that buried rows of the patch intermap are `np.nan`, and that domain indices are 0-based).

*Packaging, tests and documentation*
* `finches/__init__.py` resolved `_version.py` relative to the working directory under Read the Docs.
* `pyproject.toml` now declares the dependencies the package imports (`matplotlib`, `tqdm`, `mdtraj`, `networkx`) and no longer declares three it never imports (`pandas`, `afrc`, `soursop`); `requires-python` is 3.9, which the `list[str]` annotations already required. CI no longer installs with `--no-deps` into an environment that had none of the runtime dependencies. `docs/conf.py` mocks `mdtraj`, `networkx` and `IPython` so the API page builds.
* `test_epsilon_calculation.py` changed the working directory of the whole pytest session at import time; it now uses absolute data paths. New tests cover the residue-numbering collision, non-standard residue names, water/ligand filtering, the epsilon-scale convention of every `StructureSurface` scorer, the IDR-row aggregation, and the plotting helper.
* Documentation: the IDR:folded domain page now explains both modes, the structure-aware `StructureSurface` mode (recommended) and the legacy `FoldedDomain` patch mode, with a stage-by-stage comparison, the footprint model, a parameter table, and the GCN4:ADBD1 comparison figure showing where and why the two maps differ; the patch-mode description, which said patches came from a shortest-path graph, now describes the 12 Å straight-line ball the code actually uses. `StructureSurface` is also documented in the intermap caveats and the API reference; the "Calculating Epsilon" methods section, previously a one-line stub, now describes the calculation step by step; the per-residue profiles, automatic region extraction and InterLogos have a section in the IDR:IDR page. Every code example in the docs was checked against the current API, the stale example notebooks were fixed (`basic_uses.ipynb` used a `temp=` keyword that does not exist; `epsilon_docs.ipynb` lacked its numpy import and used the misspelt `FoldeDomain` alias; stray `sparrow` imports removed) and re-executed, and the folded-domain examples README describes all four notebooks.


**Second review pass (September 2026): docstrings, redundancy and consistency**

A second module-by-module pass focused on docstrings, duplicated code and inconsistencies between sibling functions, with every suspected bug reproduced before it was fixed. As a global check, the Mpipi-GG reference data used by the test suite regenerates bit-identically from the reviewed code. Items marked *output-changing* alter numbers or figures; everything else changes error handling, documentation, or behaviour that previously raised.

*Output-changing*
* **CALVADOS pair potentials are now zero beyond their cutoffs** (2 nm for Ashbaugh-Hatch, 4 nm for Yukawa), as in CALVADOS itself; previously both closed forms were evaluated past the cutoff (e.g. Y-Y Ashbaugh-Hatch was +0.0016 kJ/mol at 2.2 nm). Only one CALVADOS2 lookup entry changes, because W-W is the only pair whose 1-3 sigma integration window extends past 2 nm: the W-W interaction parameter goes from -2.36087829 to -2.36095710 (3.3e-5 relative). CALVADOS2 epsilons that involve tryptophan move by up to about 5e-5 relative (poly-W against itself: -43.8263 to -43.8287); nothing else changes. The CALVADOS2 test reference data was regenerated, and the regenerated files differ from the old ones only in tryptophan-containing entries.
* `build_SALT_dependent_phase_diagrams()`, `build_PH_dependent_phase_diagrams()` and `build_DIELECTRIC_dependent_phase_diagrams()` built each phase diagram from the constructor's default charge prefactor and baseline even when an explicit `prefactor` or `null_interaction_baseline` was passed, so the diagram did not match the epsilon reported alongside it (with `prefactor=0.0` a repulsive reported epsilon of +7.90 came with a diagram built from an attractive one). Each diagram is now built from its own reported epsilon. Scans with the default arguments are bit-identical.
* `FoldedDomain.calculate_idr_surface_patch_interactions()` rebuilt the object's surface neighbour graph at its `patch_radius` (12 Å) as a side effect, so `calculate_surface_epsilon()` and `calculate_IWD()` returned different values depending on whether the patch intermap had been computed first (surface epsilons on ADBD1 moved by up to 0.096). The patches are now built for that calculation only; the patch intermap itself is unchanged.
* `interaction_figure(zero_folded=True)` for a protein against RNA masked none of the protein's folded domains, because metapredict fails on the RNA and one shared `try/except` then dropped both sequences' domains. Folded domains are now predicted per sequence.
* `domain_analysis.get_column_regions()` with `penalize_opposite=True` and `criteria='greater'` counted every neutral cell as "opposite" and found no regions; `offset_threshold` is now a magnitude whose sign follows the criterion. `plot_regions_on_intermap()` rectangles are aligned to cell edges and `InteractingRegions.rectangle_info()` ends one residue later, to match `interaction_figure()`, whose cells span p to p+1 (figures change slightly).
* `protein_nucleic_vector()` and `protein_peptide_vector()` return a `(positions, values)` tuple rather than a list, like the per-residue profile methods; unpacking is unaffected.

*Bug fixes*
* `Mpipi_model('Mpipi_original')` could not be used at all: it declared U as a valid residue but ships no U parameters, so building an interaction matrix raised. The residue set is now taken from the parameters loaded (the 20 amino acids for this version), and poly(GS) scores -2.4e-5, which confirms its calibration constants.
* `calculate_sliding_epsilon()` with a negative `window_size` raised an out-of-bounds error on the Cython path and returned a NaN-filled map on the Python path, and 0 was silently rounded up to 1; any window below 1 now raises a `ValueError`. A sequence converter that changes a sequence's length now raises a clear `ValueError` instead of a broadcasting error, and `get_null_interaction_baseline()` raises if its root solve does not converge.
* `FoldedDomain.calculate_IWD()` raised a `KeyError` whenever the surface graph had more than one connected component; pairs with no over-the-surface path now contribute zero. `all_shortest_paths` went stale after `get_nearest_neighbour_res()` rebuilt the graph. A cache in `calculate_surface_epsilon()` that was never filled, and the branch that read it, are gone.
* `intermolecular_idr_matrix(null_shuffle=0)` crashed for a homotypic pair and returned an all-NaN matrix for a heterotypic one; `null_shuffle` must now be False or a positive number. `plot_rectangles` style dictionaries that set `linewidth` or `facecolor` raised a `TypeError`; they now override the defaults. `plot_multiple_phase_diagrams()` chose its axis label from the value of the reference critical temperature rather than from whether `tc_ref` was given.
* CALVADOS rejects RNA with the same clear `ValueError` on every method that takes a sequence; `epsilon_vectors()` and several others previously reported an "unknown residue".
* `analytical_fh.backend.binodal()` validated `iteration` with `assert`, which vanishes under `python -O`; it now raises a `ValueError`. Several docstring examples in `backend.py` showed wrong values (e.g. `spinodal(3, 1)` gives [0.789, 0.211], not [0.833, 0.166]) and now show verified output.
* `domain_analysis` smoothing returned the wrong number of values for intermaps one or two windows wide, and `plot_regions_on_intermap()` silently returned `None` when the region lists and colours differed in length; it now raises a `ValueError` and returns `(fig, ax)`.
* Docstring examples in the frontend that could not run (profiles on sequences too short for the default smoothing window, missing imports) or that drew attractive values in green have been corrected.

*Redundancy removed*
* The residue classes and the two elementary weighting rules now live once, in `finches.parsing_aminoacid_sequences` (`POSITIVE_RESIDUES`, `NEGATIVE_RESIDUES`, `CHARGED_RESIDUES`, `ALIPHATIC_RESIDUES`, `get_charge_window_counts()`, `charge_weight_from_counts()`, `aliphatic_weight_from_levels()`), and both the sequence masks and `StructureSurface` use them; every mask and every `StructureSurface` output was verified bit-identical. `get_aliphatic_groups()` was simplified after verifying on 37,767 sequences that its ±4 window never had an effect: the level is exactly the cluster size capped at 3, with single-residue gaps tolerated.
* `Mpipi_frontend` and `CALVADOS_frontend` no longer override `intermolecular_idr_matrix()`, `interaction_figure()`, `epsilon()` or `protein_nucleic_vector()`; every method lives on `FinchesFrontend`, and the two frontends differ only in the class attributes `supports_rna` and `default_intermap_limit`. The paired profile methods, their plotting twins and the two InterLogo functions share one implementation each, and `dms()` computes the wild-type epsilon once. The frontend code is about 500 lines shorter.
* Forcefields: one table-driven pickle loader for Mpipi, one 1-3 sigma integration routine and distance grid in `ForcefieldModel`, and one `STANDARD_AMINO_ACIDS` list. Flory-Huggins: one supercritical-chi filter, one vectorised chi scan and one chi-to-temperature conversion. `epsilon_stateless.get_matrix_epsilon_vectors()` is shared by the stateless functions and the sliding-window fallback.
* Removed unused or broken files: `finches/tests/backup_generate_data_test_epsilon_calculation.py`, `finches/tests/test_data/update_test_data.py`, `test_sequences.py`, `example_forcefield.py`, `__init__.py` and `old_data/`, `examples/folded_domains/Untitled.ipynb`, `finches/domain_decomposition/domain_testing.ipynb` (it called functions that no longer exist), the duplicate `readthedocs.yml`, `.lgtm.yml` (a defunct service), and `finches/data/calvados/calvados_residues.pickle` (no longer readable with current pandas; its one reader, a calibration notebook, now uses the dictionary pickle).

*Consistency*
* Canonical names, with the old names kept as aliases: `compute_coulomb()` (was the misspelt `compute_colulomb()`), `compute_full_potential()` on both forcefields (was `compute_full_Mpipi()` / `compute_full_calvados()`), `CALVADOS_model` (was also `calvados_model`), and `StructureSurface.write_pdb_idr_interaction()` (was `write_pdb_IDR_interaction()`).
* `ValueError` (or `FileNotFoundError`) replaces bare `Exception` throughout, `warnings.warn` replaces `print` for warnings, and mutable default arguments are `None`. `interaction_figure()` takes `vmin`, `vmax`, `disorder_1` and `disorder_2` on both frontends, all defaulting to `None` (colour limits of ±3 for Mpipi and ±7.5 for CALVADOS; disorder predicted unless the sequence is RNA). `FoldedDomain.calculate_IWD()` always returns a list. `finches` and `finches.forcefields` declare `__all__`.

*Docstrings and documentation*
* Every module, class and public function now has a numpy-style docstring checked against the code; notable corrections include the per-residue profile definitions, the epsilon being a sum over the first sequence of a mean over the second, the Flory-Huggins return conventions, the exact contents of the shipped Mpipi-GG parameters, and the phase-diagram temperature being in units of epsilon with no reference offset.
* The docs were updated for all of the above: the API page (a working quickstart, the frontend class attributes, RNA handling, and a new forcefield-models section), the phase-diagram pages (temperature units, the missing reference offset, the chi range, condition scans), the methods page (potential cutoffs, the weighting rules and where they live), the IDR:IDR and IDR:folded domain pages, and the package-data README. Every code example in the docs was executed, and every printed value in them was checked against the current code.

*Repository and tests*
* `setup.cfg` keeps only the coverage settings; ruff is configured in `pyproject.toml` (88 columns, notebooks excluded); the project URLs are filled in. CI runs on `main` and `develop` with Python 3.9-3.12, builds the Cython extension, and uses maintained actions; the test environment now contains the scientific dependencies. The docs version is read from git.
* Test modules no longer change the working directory of the pytest session at import. New tests cover the shared weighting helpers, the sliding-window and matrix paths, the frontend consolidation (including RNA rejection on every CALVADOS method), the Flory-Huggins scan consistency, and `FoldedDomain` (`finches/tests/test_folded_domain_utils.py`); the suite is at 478 tests.

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
