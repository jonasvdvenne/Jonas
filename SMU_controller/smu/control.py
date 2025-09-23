import threading, time
from math import isfinite
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from .utils import calculate_temperature, Rolling, monotonic_s, DEFAULT_ALPHA, DEFAULT_ROOM_TEMP_C
from .pulse import PulseManager

class Controller:
    """
    Hardware-agnostic control loop (PID + Pulses + Logging), designed to be
    driven from a web UI. Runs in its own thread. 
    """
    def __init__(self, smu, pid_profile, loop_dt=0.004, default_max_current=0.52, warmup_current=0.050):
        self.smu = smu
        self.pid_profile = pid_profile
        self.loop_dt = float(loop_dt)
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Parameters
        self.room_temp_c = DEFAULT_ROOM_TEMP_C
        self.alpha = pid_profile.alpha
        self.target_temp_c = 37.5
        self.temp_band = 1.0

        self.r0_measured: Optional[float] = None
        self.r0_floor = 1e-4
        self.max_current = default_max_current
        self.volt_compliance = 2.0

        # State
        self._setpoint_value = warmup_current
        self._latest_cmd = self._setpoint_value
        self._pid_int = 0.0
        self._last_temp = None
        self._dT_filt = None
        self._last_time = None
        self._setpoint_change_tsys = 0.0


        # First-pulse gating and cooldown
        self._band_entry_tsys = None
        self._pulses_started = False
        self._last_pulse_event_t = 0.0
        self._waiting_for_stability = True
        # Pulse manager and manual overrides
        self.pulse = PulseManager()
        self.pulse_paused = True       # auto state
        self.pulse_manual_override: Optional[bool] = None  # None=auto, True=force pause, False=force active

        # Filtering
        self.temp_filter = Rolling(self.pid_profile.temp_filter_window)

        # Logs (full)
        self.log_t: List[float] = []
        self.log_i: List[float] = []
        self.log_v: List[float] = []
        self.log_r: List[float] = []
        self.log_temp: List[float] = []
        self.log_pulse_flag: List[bool] = []
        # Per-pulse slopes
        self.pulse_events: List[Dict[str, Any]] = []
        self.slope_t0 = 0.00
        self.slope_t1 = 0.50

        # Metadata (frozen at start)
        self._run_meta: Optional[Dict[str, Any]] = None

        self._lock = threading.Lock()
        self._waiting_for_stability = False
        
        self.log_slope_t = []
        self.log_slope_val = []

    def set_target_temp(self, temp_c: float):
        """Update target temperature live while running."""
        self.target_temp_c = float(temp_c)
    
        # 🔹 Reset gating so pulses wait until stable again
        self._band_entry_tsys = None
        self._waiting_for_stability = True
        self.pulse_paused = True
        self._pulses_started = False
    
        # 🔹 Reset PID state
        self._pid_int = 0.0
        self._last_temp = None
        self._dT_filt = None
        self._last_time = None





    # ------------------ hardware config ------------------

    def apply_basic_settings(self, src_range_a: float, meas_i_range_a: float, meas_v_range_v: float):
        self.smu.write(":SOUR:FUNC CURR")
        self.smu.write(f":SOUR:CURR:RANG {float(src_range_a)}")
        self.smu.write(f":SENS:CURR:RANG {float(meas_i_range_a)}")
        self.smu.write(f":SENS:VOLT:RANG {float(meas_v_range_v)}")
        self.smu.write(f":SENS:VOLT:PROT {float(self.volt_compliance)}")
        self.smu.write(":SENS:VOLT:APER 0.001"); self.smu.write(":SENS:CURR:APER 0.001")
        self.smu.write(":SENS:CURR:PROT 1"); self.smu.write("SENS:REM ON")

    # ------------------ R0 measurement ------------------

    def measure_r0(self, i_r0=0.001, samples=100, dly_s=0.01):
        self.smu.write("OUTP ON"); self.smu.write(f"SOUR:CURR {i_r0:.6f}")
        vals = []
        for _ in range(int(samples)):
            self.smu.write("MEAS:CURR?"); i_meas = float(self.smu.read())
            self.smu.write("MEAS:VOLT?"); v_meas = float(self.smu.read())
            if isfinite(i_meas) and i_meas > 0:
                vals.append(v_meas / i_meas)
            time.sleep(max(dly_s, 0.002))
        self.smu.write("OUTP OFF")
        if vals:
            r0 = float(np.mean(vals))
            if isfinite(r0) and r0 > 0:
                self.r0_measured = r0
                self.r0_floor = i_r0
        return self.r0_measured

    # ------------------ lifecycle ------------------

    def start(self):
        if self._running:
            return
        if self.r0_measured is None:
            raise RuntimeError("Measure R0 before starting output.")
        self._pid_int = 0.0
        self._last_temp = None
        self._dT_filt = None
        self._last_time = None
        self._setpoint_value = max(self._setpoint_value, self.r0_floor)
        self._latest_cmd = self._setpoint_value
        self._running = True
        self._freeze_metadata()
        self.smu.write("OUTP ON")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            self.smu.write("OUTP OFF")
        except:
            pass

    # ------------------ control helpers ------------------

    def _freeze_metadata(self):
        try:
            idn = self.smu.query("*IDN?").strip()
        except Exception:
            idn = None
        self._run_meta = {
            "schema_version": "2.0.0",
            "created_utc": time.time(),
            "instrument_idn": idn,
            "r0_ohm": float(self.r0_measured) if self.r0_measured else None,
            "room_temp_c": float(self.room_temp_c),
            "alpha_per_c": float(self.alpha),
            "target_temp_c": float(self.target_temp_c),
            "max_current_a": float(self.max_current),
            "compliance_v": float(self.volt_compliance),
            "pulse_settings": {
                "enabled": bool(self.pulse.enabled),
                "power_w": float(self.pulse.power_w),
                "duration_s": float(self.pulse.duration_s),
                "period_s": float(self.pulse.period_s),
            },
            "slope_window_s": {"t_start": float(self.slope_t0), "t_end": float(self.slope_t1)},
            "pid_profile": self.pid_profile.model_dump(),
        }

    def _slew_limited(self, target_i, current_i, dt, max_slew=10.0):
        max_delta = max_slew * dt
        delta = np.clip(target_i - current_i, -max_delta, +max_delta)
        return current_i + delta

    # ------------------ main loop ------------------

    def _loop(self):
        t0 = monotonic_s()
        latest_i_cmd = self.r0_floor
        while self._running:
            t = monotonic_s() - t0
            try:
                # Read measurement
                resp = self.smu.query("MEAS:CURR?;:MEAS:VOLT?")
                parts = resp.replace(";", " ").replace(",", " ").split()
                if len(parts) >= 2:
                    i_meas = float(parts[0]); v_meas = float(parts[1])
                else:
                    self.smu.write("MEAS:CURR?"); i_meas = float(self.smu.read())
                    self.smu.write("MEAS:VOLT?"); v_meas = float(self.smu.read())
                r_meas = (v_meas / i_meas) if i_meas != 0 else np.nan

                temp = calculate_temperature(r_meas, self.r0_measured, self.alpha, self.room_temp_c)
                self.temp_filter.push(temp)
                base_temp = self.temp_filter.mean()

                # PID
                if not isfinite(base_temp):
                    self._setpoint_value = max(self._setpoint_value, self.r0_floor)
                else:
                    if self._last_time is None:
                        self._last_time = t
                    dt = max(1e-6, t - self._last_time)
                    self._last_time = t
                    err = self.target_temp_c - base_temp
                    Kp, Ki, Kd = self.pid_profile.pick_gains(abs(err))

                    kd_scale = 1.5
                    if (t - self._last_pulse_event_t) <= self.pid_profile.post_pulse_cooldown_s:
                        kd_scale = 0.01

                    # Anti-windup: only integrate when not saturated
                    if self.r0_floor < self._setpoint_value < self.max_current:
                        self._pid_int = float(np.clip(self._pid_int + err * dt, -self.pid_profile.int_clip, self.pid_profile.int_clip))

                    dT = (base_temp - (self._last_temp if self._last_temp is not None else base_temp)) / dt
                    self._last_temp = base_temp
                    alpha_f = 0.3
                    self._dT_filt = dT if self._dT_filt is None else (1 - alpha_f) * self._dT_filt + alpha_f * dT
                    d = -np.clip(self._dT_filt, -self.pid_profile.deriv_clip, self.pid_profile.deriv_clip)

                    adj = Kp * err + Ki * self._pid_int + (Kd * kd_scale) * d

                    proposed = self._setpoint_value + adj
                    if (t - self._last_pulse_event_t) <= self.pid_profile.post_pulse_cooldown_s:
                        min_allowed = self._setpoint_value - self.pid_profile.post_pulse_drop_limit
                        if proposed < min_allowed:
                            proposed = min_allowed

                    self._setpoint_value = float(np.clip(proposed, self.r0_floor, self.max_current))

                    # --- Stability gating before pulses ---
                    if self.pulse_manual_override is None:
                        if self._waiting_for_stability:
                            if isfinite(base_temp) and abs(base_temp - self.target_temp_c) <= abs(self.temp_band):
                                if self._band_entry_tsys is None:
                                    self._band_entry_tsys = time.time()
                                elif time.time() - self._band_entry_tsys >= 5.0:
                                    self._waiting_for_stability = False
                                    self.pulse_paused = False
                            else:
                                self._band_entry_tsys = None



                # Pulse state
                if self.pulse_manual_override is True:
                    effective_paused = True
                elif self.pulse_manual_override is False:
                    effective_paused = False
                else:
                    effective_paused = self.pulse_paused

                pulse_active_now = False
                i_pulse = 0.0
                if not effective_paused and self.pulse.enabled:
                    before = self.pulse.active
                    self.pulse.tick()
                    after = self.pulse.active
                    pulse_active_now = after
                    i_pulse = self.pulse.current_for_temp(
                        r0=self.r0_measured,
                        alpha=self.alpha,
                        t0=self.room_temp_c,
                        temp_c=self.target_temp_c,
                        i_max=self.max_current,
                        v_comp=self.volt_compliance
                    )
                    if after and not before:
                        self._last_pulse_event_t = t
                        self.pulse_events.append({
                            "start_time": float(t),
                            "end_time": None,
                            "slope_computed": False,
                            "slope_c_per_s": None,
                            "window_params": {"t_start": float(self.slope_t0), "t_end": float(self.slope_t1)}
                        })
                    if (not after) and before:
                        self._last_pulse_event_t = t
                        for ev in reversed(self.pulse_events):
                            if ev.get("end_time") is None:
                                ev["end_time"] = float(t)
                                break

                # Command
                baseline = self._setpoint_value
                target_i = float(np.clip(baseline + i_pulse, self.r0_floor, self.max_current))
                latest_i_cmd = self._slew_limited(target_i, latest_i_cmd, self.loop_dt, max_slew=10.0)
                self._latest_cmd = latest_i_cmd
                self.smu.write(f"SOUR:CURR {latest_i_cmd:.6f}")

                # Log
                with self._lock:
                    self.log_t.append(t)
                    self.log_i.append(i_meas)
                    self.log_v.append(v_meas)
                    self.log_r.append(r_meas)
                    self.log_temp.append(temp)
                    self.log_pulse_flag.append(bool(pulse_active_now))

                # If a pulse window finished, compute slope
                self._compute_slope_if_ready(t)
            except Exception as e:
                print("Control loop error:", e)   # 🔹 debug output
                try:
                    self.smu.write("OUTP OFF")
                except:
                    pass
                self._running = False


            time.sleep(self.loop_dt)

    def _compute_slope_if_ready(self, now_t: float):
        if not self.pulse_events:
            return
        dt0, dt1 = float(self.slope_t0), float(self.slope_t1)
        if dt1 <= dt0: return
        for ev in reversed(self.pulse_events):
            if ev.get("slope_computed", False): 
                continue
            start = ev.get("start_time", None)
            if start is None: 
                continue
            if now_t < (start + dt1): 
                continue
            # collect samples
            with self._lock:
                t_arr = np.asarray(self.log_t, dtype=float)
                temp_arr = np.asarray(self.log_temp, dtype=float)
            if t_arr.size == 0 or temp_arr.size == 0: 
                continue
            mask = (t_arr >= (start + dt0)) & (t_arr <= (start + dt1)) & np.isfinite(temp_arr)
            if np.count_nonzero(mask) < 5:
                ev["slope_computed"] = True
                ev["slope_c_per_s"] = np.nan
                continue
            tx = t_arr[mask]; ty = temp_arr[mask]
            try:
                m, b = np.polyfit(tx, ty, 1)
                slope = float(m)
            except Exception:
                slope = float('nan')
            ev["slope_computed"] = True
            ev["slope_c_per_s"] = slope
            ev["slope_window"] = {"t_start": float(dt0), "t_end": float(dt1)}
            t_mid = float((tx[0] + tx[-1]) / 2.0)
            self.log_slope_t.append(t_mid)
            self.log_slope_val.append(slope)

    # ------------------ public getters / setters ------------------

    def set_pid_profile(self, profile):
        self.pid_profile = profile
        self.alpha = profile.alpha    # 🔹 keep alpha in sync with profile
        self.temp_filter = Rolling(profile.temp_filter_window)


    def set_pulse(self, power_w, duration_s, period_s):
        self.pulse.configure(power_w, duration_s, period_s)

    def set_basic_params(self, room_temp_c, target_temp_c, temp_band,
                         max_current, volt_compliance,
                         current_range="Auto", voltage_range="Auto"):
        self.room_temp_c = room_temp_c
        # alpha always comes from profile
        self.alpha = self.pid_profile.alpha
        self.target_temp_c = target_temp_c
        self.temp_band = temp_band
        self.max_current = max_current
        self.volt_compliance = volt_compliance
    
        # 🔹 Push ranges & compliance down to SMU
        self.smu.set_ranges(current_range, voltage_range)
        try:
            self.smu.write(f":SENS:VOLT:PROT {float(self.volt_compliance)}")
        except Exception:
            pass
    
        

    def latest_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "latest_cmd_a": float(self._latest_cmd),
                "setpoint_a": float(self._setpoint_value),
                "last_temp_c": (self._last_temp if self._last_temp is not None else None),
                "n_samples": len(self.log_t),
                "pulse_active": self.pulse.active if self.pulse.enabled else False,
            }

    def snapshot(self) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[bool]]:
        with self._lock:
            return (list(self.log_t), list(self.log_i), list(self.log_v), list(self.log_r), list(self.log_temp), list(self.log_pulse_flag))

    def metadata(self) -> Dict[str, Any]:
        return self._run_meta or {}

    def export_rows(self) -> List[Dict[str, Any]]:
        t, i, v, r, temp, flag = self.snapshot()
        rows = []
        for ts_, ia, vv, rr, tc, pf in zip(t, i, v, r, temp, flag):
            rows.append({
                "t_s": float(ts_),
                "i_a": float(ia),
                "v_v": float(vv),
                "r_ohm": float(rr) if (rr is not None and np.isfinite(rr)) else None,
                "temp_c": float(tc) if (tc is not None and np.isfinite(tc)) else None,
                "pulse_active": bool(pf)
            })
        return rows

    def export_payload(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata(),
            "samples": self.export_rows(),
            "pulse_events": self._serialize_pulse_events()
        }

    def _serialize_pulse_events(self, events=None):
        events = events or self.pulse_events
        out = []
        for ev in events:
            out.append({
                "start_time_s": float(ev.get("start_time")) if ev.get("start_time") is not None else None,
                "end_time_s": float(ev.get("end_time")) if ev.get("end_time") is not None else None,
                "slope_c_per_s": (float(ev.get("slope_c_per_s"))
                                  if (ev.get("slope_c_per_s") is not None and np.isfinite(ev.get("slope_c_per_s")))
                                  else None),
                "slope_computed": bool(ev.get("slope_computed", False)),
                "slope_window_s": ev.get("slope_window") or ev.get("window_params") or {
                    "t_start": float(self.slope_t0), "t_end": float(self.slope_t1)
                }
            })
        return out
