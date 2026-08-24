# Folded-domain surface analysis examples

These examples demonstrate `finches.utils.structure_surface.StructureSurface`, which
reads a protein structure (PDB **or** mmCIF/PDBx — mdtraj handles both) and provides a
structure-aware interaction workflow:

1. **Decompose** each chain into folded domains and IDRs (via metapredict), including
   IDR residues that are missing from the coordinates.
2. **Classify** folded-domain residues as solvent-exposed (surface) or buried. SASA is
   computed on the folded domains only, so disordered tails don't occlude the surface.
3. Build a **contiguous-surface net** — a graph over surface residues where neighbours
   share a real, continuous surface (residues close in space but on opposite faces, or
   separated by the buried core, are *not* linked).
4. Score **context-aware interactions** for each surface residue against an IDR (any
   length) or another surface residue, using FINCHES energetics with the charge and
   aliphatic weighting applied over the *surface patch* rather than the linear sequence.
5. Apply a **polymer-reach** constraint (`R(n) = 5 * n^0.54` Å) to limit where an
   anchored IDR can physically reach, in cis (same chain) and trans (other chains).

## Files

| Notebook | Structure | Format | Highlights |
|----------|-----------|--------|-----------|
| `example_single_protein_p53.ipynb` | p53 (`AF-P04637-F1-model_v6.pdb`) | PDB | single chain; IDR/folded decomposition; charge-complementary surface scoring; reach |
| `example_multiprotein_npm1.ipynb` | NPM1 pentamer (`fold_npm1_pentamer_model_0.cif`) | mmCIF | 5 chains; **cross-chain** surface net; surface-vs-surface interface contacts; cis vs trans reach |

## Running

Open either notebook with Jupyter and run the cells, e.g. from this directory with
the `finches` environment active:

```bash
jupyter lab example_single_protein_p53.ipynb
jupyter lab example_multiprotein_npm1.ipynb
```

Both notebooks ship with the executed outputs embedded, so you can also read them
without re-running. Building the surface net for the NPM1 pentamer (~500 surface
residues across 5 chains) takes a little longer than the single-domain p53 example.

## What to expect

- **p53**: two large IDRs (the N-terminal transactivation/proline-rich region and the
  C-terminal regulatory region) flanking the folded core. A basic (Arg/Lys-rich) IDR
  scores most attractively against the acidic surface residues (Asp/Glu) — i.e. charge
  complementarity falls out of the FINCHES weighting.
- **NPM1 pentamer**: the oligomerization interface appears as **cross-chain** edges in
  the surface net, the strongest surface-vs-surface contacts are interfacial salt
  bridges (e.g. Lys↔Asp across chains), and an IDR anchored on one chain can reach
  surface residues on neighbouring chains (trans), not just its own (cis).

## Notes

- `StructureSurface` accepts the full per-chain sequence(s) via the `sequences=`
  argument to recover IDR residues missing from the coordinates. The AlphaFold models
  used here model the entire sequence, so this isn't needed; it matters for
  experimental structures with disordered loops absent from the density.
- The contiguous-surface occlusion test has tunable parameters
  (`occlusion_radius`, `occlusion_surround_thresh`); the defaults were calibrated on a
  small folded domain and may benefit from light tuning on very large assemblies.
