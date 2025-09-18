import sys
import time
import json
from math import isfinite

import numpy as np
import pyvisa

from PyQt5.QtCore import QThread, QObject, pyqtSignal, QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QMessageBox, QFileDialog
)
import pyqtgraph as pg


# =========================
# Utility
# =========================

def safe_float(line_edit: QLineEdit, default: float) -> float:
    try:
        return float(line_edit.text())
    except Exception:
        return default


def calculate_temperature(r, r0, alpha=0.00381, t0=22.0):
    if r0 is None or r0 <= 0 or not isfinite(r0) or not isfinite(r) or r <= 0:
        return np.nan
    return (r / r0 - 1.0) / alpha + t0


# =========================
# Pulse Manager
# =========================

class PulseManager:
    def __init__(self):
        self.enabled = False
        self.power_w = 0.0
        self.duration_s = 0.0
        self.period_s = 0.0
        self._last_pulse_time = 0.0
        self._pulse_active = False
        self._pulse_end_time = 0.0

    def configure(self, power_w, duration_s, period_s):
        self.enabled = (power_w > 0 and duration_s > 0 and period_s > 0)
        self.power_w = power_w
        self.duration_s = duration_s
        self.period_s = period_s
        self._last_pulse_time = 0.0
        self._pulse_active = False
        self._pulse_end_time = 0.0

    def get_pulse_current(self, r0: float) -> float:
        if not self.enabled or not isfinite(r0) or r0 <= 0:
            return 0.0

        now = time.time()

        # start new pulse?
        if not self._pulse_active and now - self._last_pulse_time >= self.period_s:
            self._pulse_active = True
            self._last_pulse_time = now
            self._pulse_end_time = now + self.duration_s

        # end pulse?
        if self._pulse_active and now >= self._pulse_end_time:
            self._pulse_active = False

        if self._pulse_active:
            return np.sqrt(self.power_w / r0)  # constant-current pulse sized from R0
        else:
            return 0.0

    def get_pulse_current_for_temp(self, r0: float, alpha: float, t0: float,
                                   temp_c: float, i_max: float = None, v_comp: float = None) -> float:
        """
        Compute pulse current that delivers ~constant power at a chosen temperature.
        Uses predicted resistance R(T). Also advances the pulse timing state.
        Obeys optional current and voltage clamps.
        """
        if not self.enabled or not isfinite(r0) or r0 <= 0 or not isfinite(alpha) or not isfinite(temp_c):
            return 0.0

        # --- TICK THE PULSE STATE MACHINE (start/stop based on time) ---
        _ = self.get_pulse_current(r0)   # updates _pulse_active, _last_pulse_time, _pulse_end_time

        # If we're not currently in a pulse window, no extra current
        if not self._pulse_active:
            return 0.0

        # Predict resistance at chosen temperature
        r_target = r0 * (1.0 + alpha * (temp_c - t0))
        if not isfinite(r_target) or r_target <= 0:
            return 0.0

        # Base current for desired power
        i = np.sqrt(max(self.power_w, 0.0) / r_target)

        # Respect compliance voltage (if provided)
        if isfinite(v_comp) and v_comp > 0:
            i = min(i, v_comp / r_target)

        # Respect current limit (if provided)
        if isfinite(i_max) and i_max > 0:
            i = min(i, i_max)

        return float(i)


# =========================
# Worker thread
# =========================

class SMUWorker(QObject):
    data_ready = pyqtSignal(float, float, float, float)
    finished = pyqtSignal()

    def __init__(self, smu, get_target_cmd_func,
                 max_current_clamp=0.52, min_current_floor=1e-4, max_slew_a_per_s=10.0):
        super().__init__()
        self.smu = smu
        self.get_target_cmd = get_target_cmd_func  # renamed for clarity
        self.running = False
        self.start_time = None
        self.latest_i_cmd = min_current_floor
        self.max_current = float(max_current_clamp)
        self.min_current = float(min_current_floor)
        self.max_slew = float(max_slew_a_per_s)
        self.loop_dt = 0.001  # ~50 Hz

    def _slew_limited(self, target_i, current_i, dt):
        max_delta = self.max_slew * dt
        delta = np.clip(target_i - current_i, -max_delta, +max_delta)
        return current_i + delta

    def start(self):
        self.running = True
        self.start_time = time.time()
        try:
            self.smu.write(f"SOUR:CURR {self.min_current:.6f}")
        except Exception as e:
            print("Worker init write failed:", e)

        while self.running:
            t = time.time() - self.start_time
            try:
                i_target = float(self.get_target_cmd())
                if not isfinite(i_target):
                    i_target = self.min_current
                i_target = np.clip(i_target, self.min_current, self.max_current)

                i_cmd = self._slew_limited(i_target, self.latest_i_cmd, self.loop_dt)
                self.latest_i_cmd = i_cmd

                self.smu.write(f"SOUR:CURR {i_cmd:.6f}")

                self.smu.write("MEAS:CURR?")
                i_meas = float(self.smu.read())
                self.smu.write("MEAS:VOLT?")
                v_meas = float(self.smu.read())
                r_meas = (v_meas / i_meas) if i_meas != 0 else np.nan

                self.data_ready.emit(t, i_meas, v_meas, r_meas)

            except Exception as e:
                print("Worker error:", e)
                self.running = False

            time.sleep(self.loop_dt)

        self.finished.emit()

    def stop(self):
        self.running = False


# =========================
# Main GUI
# =========================

class SMUGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keysight B2901 Temp Control (PID + Pulses + Slope)")
        self.resize(1280, 1000)

        self.smu = self._init_smu()
        self.worker = None
        self.thread = None

        self.t_buf, self.i_buf, self.v_buf, self.r_buf = [], [], [], []
        self.temp_buf, self.temp_filter_buf = [], []
        self.pulse_flag_buf = []  # per-sample pulse activity flag for exports
        self.log_t, self.log_i, self.log_v, self.log_r = [], [], [], []
        self.log_temp, self.log_pulse_flag = [], []
        # Per-pulse slope results
        self.pulse_events = []        # list of dicts per pulse (start/end/slope etc.)
        self.slope_times = []         # x for slope plot (pulse start time)
        self.slope_values = []        # y for slope plot (°C/s)

        self.r0_measured = None
        self.r0_current_used = 1e-4
        self.pid_int = 0.0
        self.pid_prev_err = 0.0

        self._default_max_current = 0.52
        self._warmup_current = 0.050
        self._temp_filter_window = 10
        self._deriv_clip = 50.0
        self._int_clip = 1000.0

        self._setpoint_value = self._warmup_current
        self._latest_cmd = self._setpoint_value  # combined baseline + pulse (NEW)

        # pulse manager + control flags
        self.pulse_manager = PulseManager()
        self._pulse_paused = True
        self._pulse_manual_override = None
        self._active_target_temp = 37.5

        # First-pulse settling and post-pulse handling
        self._band_entry_time = None      # when we entered band
        self._pulses_started = False      # has the first pulse been allowed
        self._last_pulse_event_ts = 0.0   # last pulse start/end timestamp
        self._post_pulse_cooldown_s = 1.0 # lighten derivative for this long after pulse
        self._post_pulse_drop_limit = 0.02  # A per control update (soft clamp)

        # Track pulse active state for slope windowing
        self._prev_pulse_active = False

        self._build_ui()
        self._wire_runtime()

    # ---------------- SMU init ----------------
    def _init_smu(self):
        rm = pyvisa.ResourceManager()
        for addr in rm.list_resources():
            try:
                inst = rm.open_resource(addr)
                idn = inst.query("*IDN?").strip()
                if "KEYSIGHT" in idn.upper() and "B2901" in idn.upper():
                    print(f"Connected to: {idn}")
                    inst.timeout = 5000
                    inst.write_termination = '\n'
                    inst.read_termination = '\n'
                    inst.write("*RST"); inst.write("*CLS")
                    inst.write(":SOUR:FUNC:MODE CURR")
                    inst.write(":SENS:CURR:PROT 1")
                    inst.write("SOUR:CURR 0.0001")
                    inst.write("SOUR:CURR:RANG 0.3")
                    inst.write("SENS:VOLT:PROT 2")
                    inst.write("SENS:VOLT:RANG 2")
                    inst.write("SENS:REM ON")
                    inst.write(":SENS:VOLT:APER 0.001")
                    inst.write("SENS:CURR:RANG 0.4")
                    inst.write(":SENS:CURR:APER 0.001")
                    return inst
            except Exception as e:
                print(f"Could not connect to {addr}: {e}")
        print("No Keysight B2901 found.")
        return None

    # ---------------- UI ----------------
    def _row(self, label, widget):
        row = QHBoxLayout()
        lab = QLabel(label); lab.setFixedWidth(220)
        row.addWidget(lab); row.addWidget(widget, 1)
        return row

    def _build_ui(self):
        pg.setConfigOptions(antialias=True)
        layout = QVBoxLayout()

        self.room_temp = QLineEdit("22.0")
        self.alpha = QLineEdit("0.00381")
        self.target_temp = QLineEdit("37.5")
        self.temp_band = QLineEdit("1.0")

        self.r0_current = QLineEdit("0.0010")
        self.r0_samples = QLineEdit("100")
        self.r0_delay_s = QLineEdit("0.01")

        self.max_current = QLineEdit(f"{self._default_max_current:.3f}")
        self.volt_compliance = QLineEdit("2.0")

        self.src_range = QComboBox(); self.src_range.addItems(["0.01","0.03","0.1","0.3","0.4","1.0"]); self.src_range.setCurrentText("0.3")
        self.meas_i_range = QComboBox(); self.meas_i_range.addItems(["0.01","0.03","0.1","0.3","0.4","1.0"]); self.meas_i_range.setCurrentText("0.4")
        self.meas_v_range = QComboBox(); self.meas_v_range.addItems(["0.2","2","20","200"]); self.meas_v_range.setCurrentText("2")

        # pulse params
        self.pulse_power = QLineEdit("0.0")
        self.pulse_duration = QLineEdit("0.0")
        self.pulse_period = QLineEdit("0.0")

        # slope window params (relative to pulse start)
        self.slope_t_start = QLineEdit("0.00")
        self.slope_t_end = QLineEdit("0.50")

        layout.addLayout(self._row("Room Temp (°C)", self.room_temp))
        layout.addLayout(self._row("Alpha (1/°C)", self.alpha))
        layout.addLayout(self._row("Target Temp (°C)", self.target_temp))
        layout.addLayout(self._row("Boundary Δ (°C)", self.temp_band))
        layout.addLayout(self._row("R₀ Measure Current (A)", self.r0_current))
        layout.addLayout(self._row("R₀ Samples", self.r0_samples))
        layout.addLayout(self._row("R₀ Delay (s)", self.r0_delay_s))
        layout.addLayout(self._row("Max Current Clamp (A)", self.max_current))
        layout.addLayout(self._row("Compliance Voltage (V)", self.volt_compliance))
        layout.addLayout(self._row("Source Current Range (A)", self.src_range))
        layout.addLayout(self._row("Meas Current Range (A)", self.meas_i_range))
        layout.addLayout(self._row("Meas Voltage Range (V)", self.meas_v_range))
        layout.addLayout(self._row("Pulse Power (W)", self.pulse_power))
        layout.addLayout(self._row("Pulse Duration (s)", self.pulse_duration))
        layout.addLayout(self._row("Pulse Period (s)", self.pulse_period))
        layout.addLayout(self._row("Slope Window Start (s)", self.slope_t_start))
        layout.addLayout(self._row("Slope Window End (s)", self.slope_t_end))

        self.current_setpoint = QLineEdit(f"{self._warmup_current:.6f}")
        layout.addLayout(self._row("Current Setpoint (A) [PID+Pulse]", self.current_setpoint))

        btns = QHBoxLayout()
        self.btn_update = QPushButton("Apply Settings")
        self.btn_r0 = QPushButton("Measure R₀")
        self.btn_toggle = QPushButton("Start Output"); self.btn_toggle.setCheckable(True)
        self.btn_pulse_toggle = QPushButton("Resume Pulses")
        self.btn_reset = QPushButton("Reset Plots")
        self.btn_save_csv = QPushButton("Save CSV")
        self.btn_save_json = QPushButton("Save JSON")
        for b in (self.btn_update, self.btn_r0, self.btn_toggle, self.btn_pulse_toggle, self.btn_reset, self.btn_save_csv, self.btn_save_json):
            btns.addWidget(b)
        layout.addLayout(btns)

        self.lab_r0 = QLabel("R₀: -- Ω"); self.lab_temp = QLabel("Filtered Temp: -- °C"); self.lab_iv = QLabel("I: -- | V: --")
        for lab in (self.lab_r0, self.lab_temp, self.lab_iv): lab.setStyleSheet("font-weight:bold"); layout.addWidget(lab)

        # pulse status label
        self.lab_pulse = QLabel("Pulses: Paused")
        self.lab_pulse.setStyleSheet("font-weight:bold; color: red")
        layout.addWidget(self.lab_pulse)

        # Plots
        self.plot_i = pg.PlotWidget(title="Current (A)"); self.plot_i.setBackground('w')
        self.curve_i = self.plot_i.plot([],[],pen=pg.mkPen(color='k', width=2),symbol='o', symbolSize=6, symbolBrush='k')
        layout.addWidget(self.plot_i)

        self.plot_t = pg.PlotWidget(title="Temperature (°C)"); self.plot_t.setBackground('w')
        self.curve_t = self.plot_t.plot([],[],pen=pg.mkPen(color='b', width=2),symbol='o', symbolSize=6, symbolBrush='b')
        self.line_target=pg.InfiniteLine(angle=0,pen=pg.mkPen(color='k',style=Qt.DashLine))
        self.line_upper=pg.InfiniteLine(angle=0,pen=pg.mkPen(color='r'))
        self.line_lower=pg.InfiniteLine(angle=0,pen=pg.mkPen(color='r'))
        for l in (self.line_target,self.line_upper,self.line_lower): self.plot_t.addItem(l)
        layout.addWidget(self.plot_t)

        # NEW: Pulse slope plot
        self.plot_slope = pg.PlotWidget(title="Pulse Slope (°C/s) — window relative to pulse start")
        self.plot_slope.setBackground('w')
        self.curve_slope = self.plot_slope.plot([], [], pen=None, symbol='o', symbolSize=8, symbolBrush='m')
        layout.addWidget(self.plot_slope)

        self.setLayout(layout)
        self.gui_timer=QTimer(self); self.gui_timer.setInterval(10); self.gui_timer.timeout.connect(self._on_gui_tick)

        # confirm target temp change
        self.target_temp.editingFinished.connect(self._confirm_target_change)

    def _wire_runtime(self):
        self.btn_update.clicked.connect(self._apply_smu_settings)
        self.btn_r0.clicked.connect(self._measure_r0)
        self.btn_toggle.clicked.connect(self._toggle_output)
        self.btn_pulse_toggle.clicked.connect(self._toggle_pulses)
        self.btn_reset.clicked.connect(self._reset_plots)
        self.btn_save_csv.clicked.connect(self._save_csv)
        self.btn_save_json.clicked.connect(self._save_json)
        self.current_setpoint.textEdited.connect(self._on_setpoint_edited)

    # ---------------- Confirm target temp ----------------
    def _confirm_target_change(self):
        new_target = safe_float(self.target_temp, self._active_target_temp)
        if self.worker and self.worker.running:
            reply = QMessageBox.question(
                self, "Confirm Target Change",
                f"Change target temperature to {new_target:.2f} °C?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                self.target_temp.setText(f"{self._active_target_temp:.2f}")
                return
        self._active_target_temp = new_target
        # re-gate first pulse after a setpoint change (auto mode only)
        if self._pulse_manual_override is None:
            self._pulse_paused = True
            self._pulses_started = False
            self._band_entry_time = None

    # ---------------- Toggle pulses manually ----------------
    def _toggle_pulses(self):
        if self._pulse_manual_override is None:
            self._pulse_manual_override = True
        elif self._pulse_manual_override is True:
            self._pulse_manual_override = False
        else:
            self._pulse_manual_override = True

        if self._pulse_manual_override is True:
            self._pulse_paused = True
            self.btn_pulse_toggle.setText("Resume Pulses")
            self.lab_pulse.setText("Pulses: Paused (manual)")
            self.lab_pulse.setStyleSheet("font-weight:bold; color: red")
        else:
            self._pulse_paused = False
            self.btn_pulse_toggle.setText("Pause Pulses")
            self.lab_pulse.setText("Pulses: Active (manual)")
            self.lab_pulse.setStyleSheet("font-weight:bold; color: green")

    # ---------------- Settings ----------------
    def _apply_smu_settings(self):
        if not self.smu: return
        try:
            vprot=safe_float(self.volt_compliance,2.0)
            self.smu.write(":SOUR:FUNC CURR")
            self.smu.write(f":SOUR:CURR:RANG {float(self.src_range.currentText())}")
            self.smu.write(f":SENS:CURR:RANG {float(self.meas_i_range.currentText())}")
            self.smu.write(f":SENS:VOLT:RANG {float(self.meas_v_range.currentText())}")
            self.smu.write(f":SENS:VOLT:PROT {vprot}")
            self.smu.write(":SENS:VOLT:APER 0.001"); self.smu.write(":SENS:CURR:APER 0.001")
            self.smu.write(":SENS:CURR:PROT 1"); self.smu.write("SENS:REM ON")
            print("SMU settings applied.")

            self.pulse_manager.configure(
                safe_float(self.pulse_power,0.0),
                safe_float(self.pulse_duration,0.0),
                safe_float(self.pulse_period,0.0),
            )
        except Exception as e: print("Apply settings failed:",e)

    # ---------------- R0 ----------------
    def _measure_r0(self):
        if not self.smu: return
        try:
            i_r0=safe_float(self.r0_current,0.001); n=int(safe_float(self.r0_samples,100)); dly=max(safe_float(self.r0_delay_s,0.01),0.002)
            print(f"Measuring R0 at {i_r0} A")
            self.smu.write("OUTP ON"); self.smu.write(f"SOUR:CURR {i_r0:.6f}")
            vals=[]
            for _ in range(n):
                self.smu.write("MEAS:CURR?"); i_meas=float(self.smu.read())
                self.smu.write("MEAS:VOLT?"); v_meas=float(self.smu.read())
                if isfinite(i_meas) and i_meas>0: vals.append(v_meas/i_meas)
                time.sleep(dly)
            if vals:
                r0=float(np.mean(vals))
                if isfinite(r0) and r0>0:
                    self.r0_measured=r0; self.r0_current_used=i_r0
                    self.lab_r0.setText(f"R₀: {r0:.4f} Ω (floor {i_r0:.6f} A)")
                    print(f"R0={r0:.6f} Ω, floor={i_r0:.6f} A")
                else: self.r0_measured=None
            else: self.r0_measured=None
        except Exception as e: print("R0 failed:",e)
        finally:
            try:self.smu.write("OUTP OFF")
            except:pass
            self._set_setpoint(max(self._warmup_current,self.r0_current_used))

    # ---------------- Output ----------------
    def _toggle_output(self):
        if not self.smu: return
        if self.btn_toggle.isChecked():
            if self.r0_measured is None:
                print("⚠ Measure R₀ first.")
                self.btn_toggle.setChecked(False)
                return
            self.pid_int=0.0; self.pid_prev_err=0.0
            self._set_setpoint(max(self._warmup_current,self.r0_current_used))
            max_i=safe_float(self.max_current,self._default_max_current)
            self.smu.write("OUTP ON")
            # Worker now reads the combined command (baseline + pulse)
            self.worker=SMUWorker(self.smu,lambda:getattr(self,"_latest_cmd",self._setpoint_value),max_i,self.r0_current_used,10.0)
            self.thread=QThread(); self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.start)
            self.worker.data_ready.connect(self._on_worker_data)
            self.worker.finished.connect(self.gui_timer.stop)
            self.worker.finished.connect(lambda:self.smu.write("OUTP OFF"))
            self.thread.start(); self.gui_timer.start()
            self.btn_toggle.setText("Stop Output"); print("Output started.")
        else:
            try:
                if self.worker:self.worker.stop()
                if self.thread:self.thread.quit(); self.thread.wait()
                if self.smu:self.smu.write("OUTP OFF")
            except: pass
            self.gui_timer.stop(); self.btn_toggle.setText("Start Output"); print("Output stopped.")

    # ---------------- Helpers ----------------
    def _set_setpoint(self,value:float):
        max_i=safe_float(self.max_current,self._default_max_current)
        value=float(np.clip(value,self.r0_current_used,max_i))
        self._setpoint_value=value
        self.current_setpoint.blockSignals(True); self.current_setpoint.setText(f"{value:.6f}"); self.current_setpoint.blockSignals(False)
        # keep latest command in sync when no pulse (baseline)
        self._latest_cmd = value

    def _on_setpoint_edited(self,text:str):
        try: val=float(text)
        except: return
        max_i=safe_float(self.max_current,self._default_max_current)
        self._setpoint_value=float(np.clip(val,self.r0_current_used,max_i))
        self._latest_cmd = self._setpoint_value

    # ---------------- Tick ----------------
    def _on_gui_tick(self): pass

    def _compute_pulse_slope_if_ready(self, now_t):
        """
        For the most recent pulse event that doesn't have a slope yet, check if we already
        have enough samples in [start+dt0, start+dt1] and, if so, compute slope of Temp vs Time.
        """
        if not self.pulse_events:
            return

        dt0 = safe_float(self.slope_t_start, 0.0)
        dt1 = safe_float(self.slope_t_end, 0.5)
        if dt1 <= dt0:
            return  # invalid window; ignore

        # Find the latest event without a slope computed
        for ev in reversed(self.pulse_events):
            if ev.get("slope_computed", False):
                continue
            start = ev.get("start_time", None)
            if start is None:
                continue
            # wait until end of window has passed
            if now_t < (start + dt1):
                continue
            # Collect samples in window
            t0 = start + dt0
            t1 = start + dt1
            # Use buffers (time and temperature)
            t_arr = np.asarray(self.t_buf)
            temp_arr = np.asarray(self.temp_buf)
            if t_arr.size == 0 or temp_arr.size == 0:
                continue
            mask = (t_arr >= t0) & (t_arr <= t1) & np.isfinite(temp_arr)
            if np.count_nonzero(mask) < 5:
                # not enough samples to fit robustly
                ev["slope_computed"] = True
                ev["slope_c_per_s"] = np.nan
                continue
            tx = t_arr[mask]
            ty = temp_arr[mask]
            # Linear regression: Temp = m * t + b
            try:
                m, b = np.polyfit(tx, ty, 1)
                slope = float(m)
            except Exception:
                slope = float('nan')

            ev["slope_computed"] = True
            ev["slope_c_per_s"] = slope
            ev["slope_window"] = {"t_start": float(dt0), "t_end": float(dt1)}
            # For plotting, place point at pulse start time
            self.slope_times.append(float(start))
            self.slope_values.append(slope)
            # Update plot
            self.curve_slope.setData(self.slope_times, self.slope_values)

            # Only compute one per call
            break

    def _on_worker_data(self, t, i_meas, v_meas, r_meas):
        # --- Temperature calculation & filtering ---
        t0 = safe_float(self.room_temp, 22.0)
        alpha = safe_float(self.alpha, 0.00381)
        temp = calculate_temperature(r_meas, self.r0_measured, alpha, t0)

        self.temp_filter_buf.append(temp)
        if len(self.temp_filter_buf) > self._temp_filter_window:
            self.temp_filter_buf.pop(0)
        valid = [x for x in self.temp_filter_buf if isfinite(x)]
        base_temp = float(np.mean(valid)) if valid else np.nan

        tgt = self._active_target_temp
        band = abs(safe_float(self.temp_band, 1.0))
        self.line_target.setPos(tgt); self.line_upper.setPos(tgt+band); self.line_lower.setPos(tgt-band)

        # --- PID controller with post-pulse protection ---
        if not isfinite(base_temp):
            self._set_setpoint(max(self._warmup_current, self.r0_current_used))
        else:
            if not hasattr(self, "_last_time"): self._last_time = t
            dt = max(1e-6, t - self._last_time); self._last_time = t

            err = tgt - base_temp

            # Gain scheduling
            if abs(err) > 10:
                Kp, Ki, Kd = 0.00120, 0.00002, 0.0012
            elif abs(err) > 5:
                Kp, Ki, Kd = 0.00080, 0.00002, 0.00090
            else:
                Kp, Ki, Kd = 0.00035, 0.000001, 0.00045

            # Post-pulse: soften derivative to avoid undershoot
            kd_scale = 1.0
            if (t - self._last_pulse_event_ts) <= self._post_pulse_cooldown_s:
                kd_scale = 0.05  # 20% derivative while cooling down

            # integral with anti-windup
            max_i = safe_float(self.max_current, self._default_max_current)
            if self.r0_current_used < self._setpoint_value < max_i:
                self.pid_int = float(np.clip(self.pid_int + err * dt, -self._int_clip, self._int_clip))

            # derivative on measurement with simple low-pass
            dT = (base_temp - getattr(self, "_last_temp", base_temp)) / dt
            self._last_temp = base_temp
            alpha_f = 0.3
            if not hasattr(self, "_dT_filt"): self._dT_filt = dT
            else: self._dT_filt = (1 - alpha_f) * self._dT_filt + alpha_f * dT
            d = -np.clip(self._dT_filt, -self._deriv_clip, self._deriv_clip)

            adj = Kp * err + Ki * self.pid_int + (Kd * kd_scale) * d

            # Soft clamp on negative step immediately after pulses to prevent big drops
            proposed = self._setpoint_value + adj
            if (t - self._last_pulse_event_ts) <= self._post_pulse_cooldown_s:
                min_allowed = self._setpoint_value - self._post_pulse_drop_limit
                if proposed < min_allowed:
                    proposed = min_allowed

            i_new = max(self.r0_current_used, proposed)
            self._set_setpoint(i_new)
            self.pid_prev_err = err

        # --- First-pulse gating: must be within band for ≥5 s (auto mode only) ---
        if self._pulse_manual_override is None and not self._pulses_started and isfinite(base_temp):
            if abs(base_temp - tgt) <= band:
                if self._band_entry_time is None:
                    self._band_entry_time = time.time()
                elif time.time() - self._band_entry_time >= 5.0:
                    self._pulse_paused = False
                    self._pulses_started = True
                    self.btn_pulse_toggle.setText("Pause Pulses")
            else:
                self._band_entry_time = None  # left the band -> reset timer

        # --- Pulse logic & status ---
        if self._pulse_manual_override is True:
            effective_paused = True
        elif self._pulse_manual_override is False:
            effective_paused = False
        else:
            effective_paused = self._pulse_paused

        i_pulse = 0.0
        pulse_active_now = False
        if not effective_paused and self.pulse_manager.enabled:
            before = self.pulse_manager._pulse_active
            # Size pulses for the setpoint temperature (tgt)
            i_pulse = self.pulse_manager.get_pulse_current_for_temp(
                r0=self.r0_measured,
                alpha=safe_float(self.alpha, 0.00381),
                t0=safe_float(self.room_temp, 22.0),
                temp_c=tgt,  # <-- use setpoint temp, not measured
                i_max=safe_float(self.max_current, self._default_max_current),
                v_comp=safe_float(self.volt_compliance, 2.0),
            )
            after = self.pulse_manager._pulse_active
            pulse_active_now = after

            # Detect pulse start/end to update cooldown timestamp and slope bookkeeping
            if after and not before:
                # pulse start
                self._last_pulse_event_ts = t
                self.pulse_events.append({
                    "start_time": float(t),
                    "end_time": None,
                    "slope_computed": False,
                    "slope_c_per_s": None,
                    "window_params": {
                        "t_start": safe_float(self.slope_t_start, 0.0),
                        "t_end": safe_float(self.slope_t_end, 0.5)
                    }
                })
            if (not after) and before:
                # pulse end
                self._last_pulse_event_ts = t
                # set end time on the latest open event
                for ev in reversed(self.pulse_events):
                    if ev.get("end_time") is None:
                        ev["end_time"] = float(t)
                        break

            self.lab_pulse.setText("Pulses: Active" + (" (manual)" if self._pulse_manual_override is False else ""))
            self.lab_pulse.setStyleSheet("font-weight:bold; color: green")
        else:
            self.lab_pulse.setText("Pulses: Paused" + (" (manual)" if self._pulse_manual_override is True else ""))
            self.lab_pulse.setStyleSheet("font-weight:bold; color: red")

        # baseline = PID output; final commanded current = baseline + pulse
        baseline = self._setpoint_value
        self._latest_cmd = baseline + i_pulse

        # --- Buffers & UI update ---
# --- Append to log buffers (never trimmed) ---
        self.log_t.append(t)
        self.log_i.append(i_meas)
        self.log_v.append(v_meas)
        self.log_r.append(r_meas)
        self.log_temp.append(temp)
        self.log_pulse_flag.append(bool(pulse_active_now))
        
        # --- Append to plot buffers (trimmed at 1000 for GUI) ---
        self.t_buf.append(t)
        self.i_buf.append(i_meas)
        self.v_buf.append(v_meas)
        self.r_buf.append(r_meas)
        self.temp_buf.append(temp)
        self.pulse_flag_buf.append(bool(pulse_active_now))
        
        if len(self.t_buf) > 1000:
            self.t_buf = self.t_buf[-1000:]
            self.i_buf = self.i_buf[-1000:]
            self.v_buf = self.v_buf[-1000:]
            self.r_buf = self.r_buf[-1000:]
            self.temp_buf = self.temp_buf[-1000:]
            self.pulse_flag_buf = self.pulse_flag_buf[-1000:]


        if len(self.t_buf)>1000:
            self.t_buf=self.t_buf[-1000:]
            self.i_buf=self.i_buf[-1000:]
            self.v_buf=self.v_buf[-1000:]
            self.r_buf=self.r_buf[-1000:]
            self.temp_buf=self.temp_buf[-1000:]
            self.pulse_flag_buf=self.pulse_flag_buf[-1000:]

        self.curve_i.setData(self.t_buf,self.i_buf)
        self.curve_t.setData(self.t_buf,self.temp_buf)
        self.lab_iv.setText(f"I: {i_meas:.6f} A | V: {v_meas:.6f} V")
        if isfinite(base_temp): self.lab_temp.setText(f"Filtered Temp: {base_temp:.2f} °C")
        else: self.lab_temp.setText("Filtered Temp: -- °C")

        # If a pulse window has completed, compute slope and update plot
        self._compute_pulse_slope_if_ready(t)

    # ---------------- Exports ----------------
    def _save_csv(self):
        """
        Save per-sample timeseries to CSV.
        Now includes metadata/settings in commented header rows.
        """
        if not self.t_buf:
            QMessageBox.information(self, "Save CSV", "No data to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", filter="CSV Files (*.csv)")
        if not path:
            return
        try:
            import csv
            with open(path, "w", newline="") as f:
                w = csv.writer(f)

                # ---------- Metadata header (commented with '#') ----------
                meta = {
                    "r0_ohm": float(self.r0_measured) if self.r0_measured is not None else None,
                    "room_temp_c": safe_float(self.room_temp, 22.0),
                    "alpha_per_c": safe_float(self.alpha, 0.00381),
                    "target_temp_c": float(self._active_target_temp),
                    "max_current_a": safe_float(self.max_current, self._default_max_current),
                    "compliance_v": safe_float(self.volt_compliance, 2.0),
                    "pulse_settings": {
                        "enabled": bool(self.pulse_manager.enabled),
                        "power_w": float(self.pulse_manager.power_w),
                        "duration_s": float(self.pulse_manager.duration_s),
                        "period_s": float(self.pulse_manager.period_s)
                    },
                    "slope_window": {
                        "t_start_s": safe_float(self.slope_t_start, 0.0),
                        "t_end_s": safe_float(self.slope_t_end, 0.5)
                    }
                }
                # Write each metadata entry as a header line
                f.write("# METADATA SETTINGS\n")
                for k, v in meta.items():
                    f.write(f"# {k}: {v}\n")
                f.write("# ----------------\n")

                # ---------- Column header ----------
                w.writerow([
                    "time_s", "current_a", "voltage_v", "resistance_ohm", "temperature_c",
                    "pulse_active"
                ])

                # ---------- Timeseries ----------
                for t,i,v,r,temp,flag in zip(
                    self.log_t, self.log_i, self.log_v, self.log_r,
                    self.log_temp, self.log_pulse_flag
                ):

                    w.writerow([
                        f"{float(t):.6f}",
                        f"{float(i):.9f}",
                        f"{float(v):.9f}",
                        f"{float(r) if isfinite(r) else np.nan:.9f}",
                        f"{float(temp) if isfinite(temp) else np.nan:.6f}",
                        int(bool(flag))
                    ])
            QMessageBox.information(self, "Save CSV", f"Saved CSV:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save CSV", f"Failed to save CSV:\n{e}")


    def _save_json(self):
        """
        Save full dataset to JSON, INCLUDING slope data and pulse event metadata.
        Also includes per-sample pulse_active flags (note of when pulses were happening).
        """
        if not self.t_buf:
            QMessageBox.information(self, "Save JSON", "No data to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", filter="JSON Files (*.json)")
        if not path:
            return
        try:
            payload = {
                "r0_ohm": float(self.r0_measured) if self.r0_measured is not None else None,
                "room_temp_c": safe_float(self.room_temp, 22.0),
                "alpha_per_c": safe_float(self.alpha, 0.00381),
                "target_temp_c": float(self._active_target_temp),
                "max_current_a": safe_float(self.max_current, self._default_max_current),
                "compliance_v": safe_float(self.volt_compliance, 2.0),
                "pulse_settings": {
                    "enabled": bool(self.pulse_manager.enabled),
                    "power_w": float(self.pulse_manager.power_w),
                    "duration_s": float(self.pulse_manager.duration_s),
                    "period_s": float(self.pulse_manager.period_s)
                },
                "slope_window": {
                    "t_start_s": safe_float(self.slope_t_start, 0.0),
                    "t_end_s": safe_float(self.slope_t_end, 0.5)
                },
                "samples": [
                    {
                        "t_s": float(t),
                        "i_a": float(i),
                        "v_v": float(v),
                        "r_ohm": float(r) if isfinite(r) else None,
                        "temp_c": float(temp) if isfinite(temp) else None,
                        "pulse_active": bool(flag)
                    }
                    for t,i,v,r,temp,flag in zip(
        self.log_t, self.log_i, self.log_v, self.log_r,
        self.log_temp, self.log_pulse_flag
    )
                ],
                "pulse_events": self._serialize_pulse_events()
            }
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            QMessageBox.information(self, "Save JSON", f"Saved JSON:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save JSON", f"Failed to save JSON:\n{e}")

    def _serialize_pulse_events(self):
        # Ensure we export numeric types + include any slopes already computed
        out = []
        for ev in self.pulse_events:
            out.append({
                "start_time_s": float(ev.get("start_time")) if ev.get("start_time") is not None else None,
                "end_time_s": float(ev.get("end_time")) if ev.get("end_time") is not None else None,
                "slope_c_per_s": (float(ev.get("slope_c_per_s"))
                                  if (ev.get("slope_c_per_s") is not None and isfinite(ev.get("slope_c_per_s")))
                                  else None),
                "slope_computed": bool(ev.get("slope_computed", False)),
                "slope_window_s": ev.get("slope_window") or ev.get("window_params") or {
                    "t_start": safe_float(self.slope_t_start, 0.0),
                    "t_end": safe_float(self.slope_t_end, 0.5)
                }
            })
        return out

    # ---------------- Misc ----------------
    def _reset_plots(self):
        self.t_buf.clear(); self.i_buf.clear(); self.v_buf.clear(); self.r_buf.clear()
        self.temp_buf.clear(); self.temp_filter_buf.clear(); self.pulse_flag_buf.clear()
        self.curve_i.setData([],[]); self.curve_t.setData([],[])

        self.pulse_events.clear()
        self.slope_times.clear(); self.slope_values.clear()
        self.curve_slope.setData([], [])
        self.log_t.clear(); self.log_i.clear(); self.log_v.clear(); self.log_r.clear()
        self.log_temp.clear(); self.log_pulse_flag.clear()

    def closeEvent(self,event):
        try:
            if self.worker:self.worker.stop()
            if self.thread:self.thread.quit(); self.thread.wait(1000)
            if self.smu:self.smu.write("OUTP OFF")
        except:pass
        super().closeEvent(event)


# =========================
# Run
# =========================A

if __name__=="__main__":
    app=QApplication(sys.argv); gui=SMUGUI(); gui.show(); sys.exit(app.exec_())
