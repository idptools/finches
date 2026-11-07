# Folded-domain surface analysis examples

These examples demonstrate `finches.utils.structure_surface.StructureSurface`, which reads a protein structure (PDB **or** mmCIF/PDBx, since mdtraj handles both) and provides a structure-aware interaction workflow:

1. **Decompose** each chain into folded domains and IDRs (via metapredict), including IDR residues that are missing from the coordinates.
2. **Classify** folded-domain residues as solvent-exposed (surface) or buried. SASA is computed on the folded domains only, so disordered tails don't occlude the surface.
3. Build a **contiguous-surface net**, a graph over surface residues where neighbours share a real, continuous surface (residues close in space but on opposite faces, or separated by the buried core, are *not* linked).
4. Score **context-aware interactions** for each surface residue against an IDR (any length) or another surface residue, using FINCHES energetics with the charge and aliphatic weighting applied over the *surface patch* rather than the linear sequence. Every score is on the FINCHES epsilon scale, so zero means "behaves like poly(GS)".
5. Build a **surface:IDR intermap** (`surface_vs_idr_matrix()`), the structural analogue of an IDR:IDR intermap: each surface residue is scored against every sliding window of the IDR, aggregated over the window-sized contact footprint an IDR could actually reach. `plot_surface_vs_idr_matrix()` draws it with buried residues greyed out and a colour scale set from the data.
6. Apply a **polymer-reach** constraint (`R(n) = 5 * n^0.54` Å) to limit where a tethered IDR can physically reach, in cis (same chain) and trans (other chains).

## Files

| Notebook | Structure | Format | Highlights |
|----------|-----------|--------|-----------|
| `example_single_protein_p53.ipynb` | p53 (`AF-P04637-F1-model_v6.pdb`) | PDB | single chain; IDR/folded decomposition; charge-complementary surface scoring; reach; a surface:IDR intermap of the p53 N-terminal region against the whole structure |
| `example_multiprotein_npm1.ipynb` | NPM1 pentamer (`fold_npm1_pentamer_model_0.cif`) | mmCIF | 5 chains; **cross-chain** surface net; surface-vs-surface interface contacts; cis vs trans reach |
| `gcn4_adb1/gcn4_abd1_structure_surface_v1.ipynb` | ADBD1 (`gcn4_adb1/ADBD1.pdb`) vs the GCN4 N-terminal IDR | PDB | the full surface:IDR intermap story: trans and cis maps, the contact footprint, why the map is smooth and on the IDR:IDR scale |
| `gcn4_adb1/gcn4_abd1_v1.ipynb` | ADBD1 vs the GCN4 N-terminal IDR | PDB | the same map built with the older `FoldedDomain` patch approach, for comparison |

## Running

Open a notebook with Jupyter and run the cells, e.g. from this directory with the `finches` environment active:

```bash
jupyter lab example_single_protein_p53.ipynb
jupyter lab example_multiprotein_npm1.ipynb
```

The notebooks ship with executed outputs embedded, so you can also read them without re-running. Building the surface net for the NPM1 pentamer (~500 surface residues across 5 chains) takes a little longer than the single-domain examples.

## What to expect

- **p53**: two large IDRs (the N-terminal transactivation/proline-rich region and the C-terminal regulatory region) flanking the folded core. A basic (Arg/Lys-rich) IDR scores most attractively against the acidic surface residues (Asp/Glu), i.e. charge complementarity falls out of the FINCHES weighting.
- **NPM1 pentamer**: the oligomerization interface appears as **cross-chain** edges in the surface net, the strongest surface-vs-surface contacts are interfacial salt bridges (e.g. Lys-Asp across chains), and an IDR anchored on one chain can reach surface residues on neighbouring chains (trans), not just its own (cis).
- **GCN4 vs ADBD1**: the acidic activation-domain-like stretches of the GCN4 IDR light up as attractive stripes against the basic face of ADBD1, and the map is smooth across neighbouring surface residues because touching residues see almost the same window.

## Notes

- `StructureSurface` accepts the full per-chain sequence(s) via the `sequences=` argument to recover IDR residues missing from the coordinates. The AlphaFold models used here model the entire sequence, so this isn't needed; it matters for experimental structures with disordered loops absent from the density.
- Residue names that are not one of the twenty standard amino acids are translated where they have an obvious equivalent (HID/HIE/HIP to H, CYX to C, MSE to M and so on); anything else is kept as a glycine placeholder with a warning.
- The contiguous-surface occlusion test has tunable parameters (`occlusion_radius`, `occlusion_surround_thresh`); the defaults were calibrated on a small folded domain and may benefit from light tuning on very large assemblies.
- The polymer-reach parameters, `contact_radius` and the surface-classification thresholds are physically motivated defaults rather than fitted values; see the warning at the top of `structure_surface.py`.
