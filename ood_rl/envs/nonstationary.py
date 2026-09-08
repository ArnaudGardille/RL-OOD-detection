"""Non-stationary dynamics: intra-trajectory parameter shifts.

Phase 0 ships a minimal single-change-point schedule. The key idea: gymnax passes
``params`` to ``step`` at every call, so a dynamics shift is just feeding different
``params`` after the change-point — fully jittable, no fragile attribute injection.
The recorded ``ground_truth_changepoint`` is what detection-delay metrics will be
measured against in Phase 1.
"""

from dataclasses import dataclass

import jax


@dataclass
class ShiftSchedule:
    """A single dynamics change-point at step ``t_change``."""

    t_change: int
    params_before: object
    params_after: object

    @property
    def ground_truth_changepoint(self) -> int:
        return self.t_change

    def params_at(self, t):
        """Return the active EnvParams at (possibly traced) step index ``t``."""
        return jax.lax.cond(
            t < self.t_change,
            lambda: self.params_before,
            lambda: self.params_after,
        )
