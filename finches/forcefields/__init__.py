"""
Forcefield models that FINCHES builds its residue-residue interactions from.

* mpipi: the Mpipi forcefield (Mpipi_model), including the Mpipi-GG parameters.
* calvados: the CALVADOS forcefield (CALVADOS_model).
* model_base: the ForcefieldModel base class every model inherits from.
"""

from . import calvados, model_base, mpipi

__all__ = ["calvados", "model_base", "mpipi"]
