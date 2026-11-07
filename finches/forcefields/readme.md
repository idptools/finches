# Forcefield models

FINCHES does not simulate anything. It takes the pair potentials of an existing coarse-grained forcefield and turns each residue pair into a single number, the interaction parameter, which is the integral of the pair potential between one and three sigma. Everything downstream (epsilon values, intermaps, phase diagrams) is built from the table of those numbers. This directory holds the forcefield models that provide the potentials.

| Module | Class | Versions | Residues | Conditions |
|---|---|---|---|---|
| `mpipi.py` | `Mpipi_model` | `Mpipi_GGv1` (default), `Mpipi_original`, `OLD_Mpipi_GGv1` (legacy) | 20 amino acids, plus U for the GGv1 parameters | `salt`, `dielectric` |
| `calvados.py` | `CALVADOS_model` | `CALVADOS2` (default), `CALVADOS1` (uncalibrated) | 20 amino acids | `salt`, `pH`, `temperature` |
| `model_base.py` | `ForcefieldModel` | base class | | |

Both models take distances in Angstroms and return energies in kJ/mol. The stand-alone CALVADOS potentials (`ashbaugh_hatch()`, `yukawa()`, `compute_calvados_energy()`) work in nanometres, like CALVADOS itself; the `CALVADOS_model` methods convert. Every model evaluates its full pair potential through `compute_full_potential()`; the older model-specific names (`compute_full_Mpipi()`, `compute_full_calvados()`) still work.

The parameter files live in `finches/data/Mpipi` and `finches/data/calvados`, and the notebooks used to calibrate the CALVADOS2 constants against Mpipi are in `calibration/`.

## Implementing a new forcefield model

A model is a subclass of `ForcefieldModel` that does three things.

1. **Calls the base constructor** with its version name, the solution conditions, its residue groups and the list of conditions its energies actually depend on. Residue groups are a list of lists: residues in one group may appear together in a sequence, residues from different groups may not (for example `[[20 amino acids], ["U"]]` for proteins and poly-U RNA).

2. **Sets `self.CONFIGS`** to a dictionary with its two calibration constants, `charge_prefactor` (between 0 and 1, the strength of the local charge weighting) and `null_interaction_baseline` (the pair interaction that counts as "no net interaction"). Store a constant you have not calibrated yet as `np.nan`. The baseline can be calibrated so that poly(GS) scores zero against itself by constructing `InteractionMatrixConstructor(model, compute_forcefield_dependencies=True)`, which calls `finches.data.forcefield_dependencies.get_null_interaction_baseline()`. The charge prefactor needs a judgement call; see `calibration/prefactor_calibration.ipynb` for how the existing values were chosen.

3. **Implements `compute_interaction_parameter(residue_1, residue_2, r=None, dielectric=None, salt=None)`**, returning the 5-tuple `(interaction_parameter, potential, i_sigma, i_3sigma, r)`. The usual pattern is to evaluate your potential on `self._get_default_distance_array()` (0.1 to 30 Angstroms in 0.01 Angstrom steps) and hand it to `self._integrate_sigma_window(potential, r, sigma)`, which returns the integral and the two indices. A missing residue pair must raise a `KeyError`, which is what the `InteractionMatrixConstructor` looks for. Implementing `compute_full_potential()` as well is optional but lets the potential be plotted through the same name as every other model.

```python
import numpy as np
from finches.forcefields.model_base import ForcefieldModel, STANDARD_AMINO_ACIDS


class MyModel(ForcefieldModel):
    def __init__(self, salt=0.150):
        super().__init__(
            version="MyModel_v1",
            dielectric=80.0,
            salt=salt,
            temperature=300,
            pH=7.4,
            all_residues_types=[list(STANDARD_AMINO_ACIDS)],
            conditions=["salt"],
        )
        self.CONFIGS = {"charge_prefactor": 0.2, "null_interaction_baseline": np.nan}
        self.SIGMA_ALL = ...  # [R1][R2] -> sigma in Angstroms

    def compute_full_potential(self, residue_1, residue_2, r, dielectric=None, salt=None):
        ...  # energy in kJ/mol at distance(s) r in Angstroms

    def compute_interaction_parameter(self, residue_1, residue_2, r=None, dielectric=None, salt=None):
        r = self._get_default_distance_array() if r is None else np.asarray(r, dtype=float)
        potential = self.compute_full_potential(residue_1, residue_2, r, salt=salt)
        value, s1, s3 = self._integrate_sigma_window(potential, r, self.SIGMA_ALL[residue_1][residue_2])
        return value, potential, s1, s3, r
```

The model is then used like any other:

```python
from finches.epsilon_calculation import InteractionMatrixConstructor

imc = InteractionMatrixConstructor(MyModel(), compute_forcefield_dependencies=True)
imc.calculate_epsilon_value("GS" * 20, "GS" * 20)  # ~0 by construction
```

If your model's energies depend on conditions, list them in `conditions` so the condition-scanning functions in `finches.epsilon_to_FHtheory` can vary them. Those functions assign to the attribute (for example `model.salt = 0.3`) and then rebuild the lookup table, so the model has to read the attribute when it evaluates the potential rather than caching values that depend on it at construction; `CALVADOS_model._sync_conditions()` shows one way to handle parameters that are expensive to recompute.
