
---

## `dlsm/__init__.py`

```python
"""DLSM: Dynamic Latent-State Model for student engagement."""

from dlsm.core.parameters import DLSParams
from dlsm.core.state import DLSMState, Status
from dlsm.core.dynamics import Dynamics
from dlsm.core.observations import ObservationModel
from dlsm.core.constraints import project_ordered_cone
from dlsm.models.dlsm import DLSM
from dlsm.inference.particle_filter import ParticleFilter, FilterResult
from dlsm.inference.pmcmc import PMCMC, PMCMCResult
from dlsm.simulation.simulator import Simulator, SimulatedData
from dlsm.evaluation import metrics
from dlsm.viz import plots

__version__ = "0.1.0"

__all__ = [
    "DLSParams",
    "DLSMState",
    "Status",
    "Dynamics",
    "ObservationModel",
    "project_ordered_cone",
    "DLSM",
    "ParticleFilter",
    "FilterResult",
    "PMCMC",
    "PMCMCResult",
    "Simulator",
    "SimulatedData",
    "metrics",
    "plots",
]