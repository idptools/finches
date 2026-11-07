# Analytical Flory-Huggins theory (`finches.analytical_fh`)

This subpackage implements the analytical solution to the two-component Flory-Huggins model described in

> Qian, D., Michaels, T. C. T., & Knowles, T. P. J. (2022). *Analytical Solution to the Flory-Huggins Model.* J. Phys. Chem. Lett. 13(33), 7853–7860. https://doi.org/10.1021/acs.jpclett.2c01986

It lets you compute spinodal and binodal concentrations for a homopolymer of length *N* in solvent, given a Flory interaction parameter χ, without any numerical root finding. The spinodal is exact and closed-form; the binodal is available either as a closed-form analytical approximation (the main result of the paper), a self-consistent Newton–Raphson iteration, or the Ginzburg–Landau expansion about the critical point.

## Layout

| File | What it is |
|---|---|
| `backend.py` | The original code from the Knowles lab ([KnowlesLab-Cambridge/FloryHuggins](https://github.com/KnowlesLab-Cambridge/FloryHuggins), originally `FH.py`), with detailed docstrings added, a handful of boundary-case bugs fixed, and the duplicated scalar and array code paths merged so they cannot disagree. Provides `critical()`, `spinodal()`, `GL_binodal()`, `binodal()` and `analytic_binodal()`. |
| `floryhuggins.py` | User-facing wrappers written by us: `calculate_binodal()` and `calculate_spinodal()`. These evaluate the solver on a grid of χ for a polymer of length *L* in one vectorized call and return lists ready for plotting. This is what `finches.epsilon_to_FHtheory` uses. |

## The model

The Flory-Huggins free energy density per lattice site is

```
f(φ) = (φ/N) ln φ + (1 − φ) ln(1 − φ) + χ φ (1 − φ)
```

where φ is the polymer volume fraction, *N* is the number of lattice sites one polymer occupies, and χ is the Flory interaction parameter. Three quantities follow from this:

* **Critical point** (`critical(N)`): χ_c = ½(1 + 1/√N)², φ_c = 1/(1 + √N). Phase separation is only possible for χ > χ_c. For long polymers χ_c → 0.5 and φ_c → 0.
* **Spinodal** (`spinodal(chi, N)`): where f''(φ) = 0. Inside the spinodal the mixed state is locally unstable. The dilute spinodal branch only scales as a power law in χ (~1/(2χN)).
* **Binodal** (`analytic_binodal(chi, N)` / `binodal(chi, N)`): the coexistence curve from the common-tangent construction. The dilute binodal branch scales *exponentially* in Nχ, which is why dilute-phase concentrations span so many orders of magnitude.

All backend functions accept either a single χ or an array of χ values. For a scalar, `spinodal` and `analytic_binodal` return `[φ_dense, φ_dilute]` while `GL_binodal` and `binodal` return `[φ_dense, φ_dilute, χ]`; for an array every function returns `[φ_dense_array, φ_dilute_array, χ_array]`, having silently dropped any χ ≤ χ_c. A scalar χ ≤ χ_c raises `ValueError`.

## Which binodal should I use?

* **`analytic_binodal`** – closed-form (eq. 34 for N = 1, eq. 36 otherwise). No iteration, no overflow, valid across the entire two-phase region. Use this unless you have a reason not to. It is an approximation, though, and a looser one than it looks on a log-scale plot. Checked against the exact common-tangent solution: for N ≤ 100 the dense branch is within ~1–8% across the two-phase region, but for N ~ 1000 it is only within ~4% right at χ_c and drifts to 25–30% by χ ≈ 1.3 χ_c (40–50% for N ~ 5000). The dilute branch is exponentially small and its *relative* error is large everywhere except very close to χ_c: 10–30% near χ_c, and orders of magnitude at large Nχ. It captures the exp(−Nχ) scaling of the dilute branch, not the prefactor. If you need the actual coexistence concentrations use `binodal`, which converges to the exact answer wherever it does not overflow.
* **`binodal`** – self-consistent iteration seeded from the Ginzburg–Landau guess, with a Newton–Raphson accelerated map by default (`UseImprovedMap=True`, `iteration=5`). Converges to the exact common-tangent answer (equal chemical potential and osmotic pressure in the two phases) to machine precision in those 5 iterations everywhere we have checked, but for large Nχ the intermediate exponentials overflow and you get `NaN`, and where the dilute branch is below about 1e-90 the last Newton step can land on a tiny negative number (for example −2e-99 for N = 5000, χ = 0.7). With fewer iterations this happens much earlier (−5e-26 for N = 100, χ = 1.5 after 3 iterations), so do not lower `iteration` below the default.
* **`GL_binodal`** – Ginzburg–Landau expansion about the critical point. Only meaningful for χ very close to χ_c; away from it the dilute branch goes negative and the dense branch exceeds 1. Included for completeness and because it seeds `binodal`.

## Building a phase diagram from temperature, a per-bead χ, and polymer length

This is the workflow the rest of finches is built around, so it is worth spelling out. The three inputs are:

1. **Polymer length, N** – the number of lattice sites one chain occupies. For a coarse-grained model with one bead per residue this is the number of residues (this is what `epsilon_to_FHtheory` does). Strictly speaking a residue is larger than a water-sized lattice site, so N is a lower bound, but the phase-diagram shape is insensitive to this at the level of accuracy Flory-Huggins offers.
2. **A per-bead Flory χ, or equivalently a per-bead contact energy ε.** In Flory-Huggins theory χ is defined *per lattice site*, so the χ for a bead is exactly the χ you feed into these functions. There is no per-chain rescaling; the chain length enters only through N. If you start from a per-chain interaction energy (e.g. the finches ε for a whole sequence) you must divide by the length first to get a per-bead value, which is what `epsilon_to_FHtheory.epsilon_to_phase_diagram` does (`delta_eps = -epsilon / len(seq)`; note the sign flip, since finches ε is negative for attraction whereas χ is positive for attraction).
3. **Temperature, T.** Temperature and χ are not independent: in the simplest (purely enthalpic) picture

   ```
   χ(T) = ε / (kB T)
   ```

   where ε is the per-bead contact energy. If ε is expressed in units of kB (i.e. in Kelvin) this is just `chi = eps_K / T`. The inverse, `T = eps_K / chi`, converts any χ axis into a temperature axis.

So temperature is not a separate knob: you pick a per-bead ε, and temperature selects a χ. Two things you might want follow directly.

### (a) Coexisting concentrations at a single temperature

```python
from finches.analytical_fh import backend as FH

eps_K = 200.0   # per-bead contact energy, in Kelvin (i.e. eps / kB)
N     = 100     # polymer length in beads
T     = 300.0   # temperature in K

chi = eps_K / T                     # 0.667
phi_c, chi_c = FH.critical(N)       # 0.0909, 0.605

if chi <= chi_c:
    print("one phase at this temperature")
else:
    phi_dense, phi_dilute = FH.analytic_binodal(chi, N)   # 0.301, 0.00407
    phi_sp_high, phi_sp_low = FH.spinodal(chi, N)
```

The critical temperature is simply `T_c = eps_K / chi_c` (330.6 K in this example); above it the system is one phase at every concentration.

### (b) A full T-vs-φ phase diagram

Scan χ over the two-phase region and then map each χ back to a temperature.

```python
import numpy as np
from finches.analytical_fh import floryhuggins

eps_K = 200.0
N     = 100

# chi_min=0.5 is safe: chi_c >= 0.5 for every N, so sub-critical chi are
# dropped automatically. chi_max sets how low in temperature you go.
chis, dilute, dense, phi_c, chi_c = floryhuggins.calculate_binodal(
    N, mode="analytic_binodal", chi_min=0.5, chi_max=1.5, n_points=2000
)
chis_s, sp_low, sp_high, _, _ = floryhuggins.calculate_spinodal(
    N, chi_min=0.5, chi_max=1.5, n_points=2000
)

# chi -> T
T_bin = eps_K / np.array(chis)
T_sp  = eps_K / np.array(chis_s)
T_c   = eps_K / chi_c

import matplotlib.pyplot as plt
plt.plot(dilute, T_bin, "k-", label="binodal")
plt.plot(dense,  T_bin, "k-")
plt.plot(sp_low,  T_sp, "k--", label="spinodal")
plt.plot(sp_high, T_sp, "k--")
plt.plot(phi_c, T_c, "ro", label="critical point")
plt.xscale("log")
plt.xlabel("volume fraction φ")
plt.ylabel("T (K)")
plt.legend()
```

Notes on this:

* `calculate_binodal` returns `(chis, dilute, dense, phi_c, chi_c)`; the lists contain only the χ values that were above χ_c, so they can be shorter than `n_points`. `chi_max` is exclusive.
* The χ grid is uniform, so the T grid is *not* – you get many points near T_c and sparse points at low T. If you want a uniform temperature grid, build your own T array, convert to χ with `chi = eps_K / T`, and call `FH.analytic_binodal(chi_array, N)` directly (remembering it drops χ ≤ χ_c and returns the surviving χ as the third row).
* Because the dilute branch is exponential in Nχ, a log x-axis is essentially mandatory for anything but very short chains.
* If ε varies with temperature (an entropic component, `χ = A + B/T`), just replace the `chi = eps_K / T` line; nothing downstream changes.
* To convert φ to a molar concentration you need a bead volume; for one-bead-per-residue models finches uses the conversions in `finches.epsilon_to_FHtheory`.

## Numerical caveats

* The array code paths keep only χ **strictly** greater than χ_c; a scalar χ ≤ χ_c raises `ValueError`. At χ = χ_c exactly, both binodal and spinodal collapse to φ_c, which you can get from `critical()`.
* `binodal()` raises `ValueError` for N < 1 or a negative `iteration`. For very large Nχ prefer `analytic_binodal` (see above).
* `calculate_binodal()` raises `ValueError` for an unknown `mode`, and returns empty lists (plus the critical point) when the whole χ grid is sub-critical.
* `GL_binodal` will happily return unphysical φ < 0 or φ > 1 — that is the approximation, not a bug.

## Original notes (2022-08-25)

Python library for calculating Flory-Huggins spinodal, approximate (Ginzburg-Landau) binodal and binodal (using the self-consistent approach). A one-line summary of each function is printed by `backend.help()`; the docstrings are the full documentation.
