import numpy as np
from math import isfinite
from .utils import monotonic_s

class PulseManager:
    """
    Pulse scheduler sized for constant power at a chosen temperature.
    State machine runs on monotonic time.
    """
    def __init__(self):
        self.enabled = False
        self.power_w = 0.0
        self.duration_s = 0.0
        self.period_s = 0.0
        self._pulse_active = False
        self._last_pulse_start = 0.0
        self._pulse_end = 0.0

    def configure(self, power_w, duration_s, period_s):
        self.enabled = (power_w > 0 and duration_s > 0 and period_s > 0)
        self.power_w = float(max(0.0, power_w))
        self.duration_s = float(max(0.0, duration_s))
        self.period_s = float(max(0.0, period_s))
        self._pulse_active = False
        self._last_pulse_start = 0.0
        self._pulse_end = 0.0

    def tick(self):
        """Advance state according to time and return active flag."""
        if not self.enabled:
            self._pulse_active = False
            return False
        now = monotonic_s()
        if not self._pulse_active:
            if (now - self._last_pulse_start) >= self.period_s:
                self._pulse_active = True
                self._last_pulse_start = now
                self._pulse_end = now + self.duration_s
        else:
            if now >= self._pulse_end:
                self._pulse_active = False
        return self._pulse_active

    @property
    def active(self):
        return self._pulse_active

    def current_for_temp(self, r0: float, alpha: float, t0: float,
                         temp_c: float, i_max: float = None, v_comp: float = None) -> float:
        """
        Compute pulse current that delivers ~constant power at a chosen temperature.
        Does not change timing; call tick() first to update active state.
        """
        if not self.enabled or not self._pulse_active:
            return 0.0
        if not (isfinite(r0) and r0 > 0 and isfinite(alpha) and isfinite(temp_c)):
            return 0.0

        r_target = r0 * (1.0 + alpha * (temp_c - t0))
        if not isfinite(r_target) or r_target <= 0:
            return 0.0

        i = np.sqrt(max(self.power_w, 0.0) / r_target)
        if v_comp and isfinite(v_comp) and v_comp > 0:
            i = min(i, v_comp / r_target)
        if i_max and isfinite(i_max) and i_max > 0:
            i = min(i, i_max)
        return float(i)
