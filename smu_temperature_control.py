import sys
import time
from math import isfinite

import numpy as np
import pyvisa

from PyQt5.QtCore import QThread, QObject, pyqtSignal, QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox
)
import pyqtgraph as pg


# =========================
# Utility / domain helpers
# =========================

def safe_float(line_edit: QLineEdit, default: float) -> float:
    try:
        return float(line_edit.text())
    except Exception:
        return default


def calculate_temperature(r, r0, alpha=0.00381, t0=22.0):
    """PT-type linearized temp estimate from resistance."""
    if r0 is None or r0 <= 0 or not isfinite(r0) or not isfinite(r) or r <= 0:
        return np.nan
    return (r / r0 - 1.0) / alpha + t0


# =========================
# VISA worker thread
# =========================

class SMUWorker(QObject):
    """
    Runs in its own thread:
      - Sets source current (with clamp & slew limit).
      - Reads measured current & voltage.
      - Computes resistance.
      - Emits tuples for GUI.
    """
    data_ready = pyqtSignal(float, float, float, float)  # t, I_meas, V, R
    finished = pyqtSignal()

    def __init__(self, smu, get_target_current_func, max_current_clamp=0.52, max_slew_a_per_s=10.0):
        super().__init__()
        self.smu = smu
        self.get_target_current = get_target_current_func  # callable returning desired I (A)
        self.running = False
        self.start_time = None
        self.latest_i_cmd = 0.0
        self.max_current = float(max_current_clamp)
        self.max_slew = float(max_slew_a_per_s)  # A/s
        self.loop_dt = 0.02   # ~50 Hz I/O loop (fits 10 ms I & 10 ms V apertures comfortably)

    def _slew_limited(self, target_i, current_i, dt):
        """Limit current change per loop to avoid big steps at the SMU output."""
        max_delta = self.max_slew * dt
        delta = np.clip(target_i - current_i, -max_delta, +max_delta)
        return current_i + delta

    def start(self):
        self.running = True
        self.start_time = time.time()

        # Ensure a defined small starting level at the instrument side
        try:
            self.smu.write(f"SOUR:CURR {0.0001:.6f}")
        except Exception as e:
            print("Worker init write failed:", e)

        while self.running:
            t = time.time() - self.start_time
            try:
                # 1) Fetch target from GUI (PID sets it); clamp it
                i_target = float(self.get_target_current())
                if not isfinite(i_target):
                    i_target = 0.0
                i_target = np.clip(i_target, 0.0, self.max_current)

                # 2) Slew-limit the commanded current we send to the SMU
                i_cmd = self._slew_limited(i_target, self.latest_i_cmd, self.loop_dt)
                self.latest_i_cmd = i_cmd

                # 3) Apply source current
                self.smu.write(f"SOUR:CURR {i_cmd:.6f}")

                # 4) Read measured I & V (use measured values for R)
                #    Using one-shot queries keeps timing simple and stable on B2901.
                self.smu.write("MEAS:CURR?")
                i_meas = float(self.smu.read())

                self.smu.write("MEAS:VOLT?")
                v_meas = float(self.smu.read())

                r_meas = (v_meas / i_meas) if i_meas != 0 else np.nan

                self.data_ready.emit(t, i_meas, v_meas, r_meas)

            except Exception as e:
                print("Worker error:", e)
                # break the loop; GUI will handle cleanup
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
        self.setWindowTitle("Keysight B2901 Temperature Control (4-wire, PID)")
        self.resize(1280, 800)

        # Instrument & state
        self.smu = self._init_smu()
        self.worker = None
        self.thread = None

        # Data buffers
        self.t_buf, self.i_buf, self.v_buf, self.r_buf = [], [], [], []
        self.temp_buf, self.temp_filter_buf = [], []

        # Control / PID state
        self.r0_measured = None
        self.pid_int = 0.0
        self.pid_prev_err = 0.0

        # GUI / control defaults
        self._default_max_current = 0.52      # A
        self._warmup_current = 0.050          # A
        self._temp_filter_window = 20         # samples for simple moving average
        self._deriv_clip = 50.0               # °C/s cap for D-term
        self._int_clip = 1000.0               # anti-windup clamp

        self._build_ui()
        self._wire_runtime()

        # Thread-safe shadow setpoint (worker reads this, not the widget)
        self._setpoint_value = self._warmup_current

    # ---------------- SMU init / config ----------------

    def _init_smu(self):
        rm = pyvisa.ResourceManager()
        for addr in rm.list_resources():
            try:
                inst = rm.open_resource(addr)
                idn = inst.query("*IDN?").strip()
                if "KEYSIGHT" in idn.upper() and any(m in idn.upper() for m in ["B2901A", "B2901B"]):
                    print(f"Connected to: {idn}")

                    # VISA settings
                    inst.timeout = 5000
                    inst.write_termination = '\n'
                    inst.read_termination = '\n'

                    # Known-good configuration sequence
                    inst.write("*RST")
                    inst.write("*CLS")
                    inst.write(":SOUR:FUNC:MODE CURR")

                    # Ranges & compliance (manual, no auto)
                    inst.write(":SENS:CURR:PROT 1")     # measurement current protect (harmless here)
                    inst.write("SOUR:CURR 0.0001")       # small starting source
                    inst.write("SOUR:CURR:RANG 0.3")     # source current range
                    inst.write("SENS:VOLT:PROT 2")       # voltage compliance
                    inst.write("SENS:VOLT:RANG 2")       # measure V range
                    inst.write("SENS:REM ON")            # 4-wire sense
                    inst.write(":SENS:VOLT:APER 0.01")   # aperture ~10ms
                    inst.write("SENS:CURR:RANG 0.4")     # measure I range
                    inst.write(":SENS:CURR:APER 0.01")

                    return inst
            except Exception as e:
                print(f"Could not connect to {addr}: {e}")
        print("No Keysight B2901 found.")
        return None

    # ---------------- UI helpers ----------------

    def _row(self, label_text, widget):
        row = QHBoxLayout()
        lab = QLabel(label_text)
        lab.setFixedWidth(220)
        row.addWidget(lab)
        row.addWidget(widget, 1)
        return row

    def _build_ui(self):
        pg.setConfigOptions(antialias=True)

        layout = QVBoxLayout()

        # --- Config & targets ---
        self.room_temp = QLineEdit("22.0")
        self.alpha = QLineEdit("0.00381")
        self.target_temp = QLineEdit("37.5")
        self.temp_band = QLineEdit("1.0")

        self.r0_current = QLineEdit("0.005")     # 5 mA default for R0 measurement
        self.r0_samples = QLineEdit("100")
        self.r0_delay_s = QLineEdit("0.01")

        self.max_current = QLineEdit(f"{self._default_max_current:.3f}")
        self.volt_compliance = QLineEdit("2.0")

        self.src_range = QComboBox(); self.src_range.addItems(["0.01","0.03","0.1","0.3","0.4","1.0"])
        self.src_range.setCurrentText("0.3")
        self.meas_i_range = QComboBox(); self.meas_i_range.addItems(["0.01","0.03","0.1","0.3","0.4","1.0"])
        self.meas_i_range.setCurrentText("0.4")
        self.meas_v_range = QComboBox(); self.meas_v_range.addItems(["0.2","2","20","200"])
        self.meas_v_range.setCurrentText("2")

        layout.addLayout(self._row("Room Temperature (°C)", self.room_temp))
        layout.addLayout(self._row("Alpha (1/°C)", self.alpha))
        layout.addLayout(self._row("Target Temperature (°C)", self.target_temp))
        layout.addLayout(self._row("Boundary Δ (°C)", self.temp_band))

        layout.addLayout(self._row("R₀ Measure Current (A)", self.r0_current))
        layout.addLayout(self._row("R₀ Samples", self.r0_samples))
        layout.addLayout(self._row("R₀ Sample Delay (s)", self.r0_delay_s))

        layout.addLayout(self._row("Max Current Clamp (A)", self.max_current))
        layout.addLayout(self._row("Compliance Voltage (V)", self.volt_compliance))
        layout.addLayout(self._row("Source Current Range (A)", self.src_range))
        layout.addLayout(self._row("Measure Current Range (A)", self.meas_i_range))
        layout.addLayout(self._row("Measure Voltage Range (V)", self.meas_v_range))

        # --- PID control output (human-visible + editable) ---
        self.current_setpoint = QLineEdit(f"{self._warmup_current:.6f}")
        layout.addLayout(self._row("Current Setpoint (A) [PID output]", self.current_setpoint))

        # --- Buttons ---
        btn_row = QHBoxLayout()
        self.btn_update = QPushButton("Apply SMU Settings")
        self.btn_r0 = QPushButton("Measure R₀")
        self.btn_toggle = QPushButton("Start Output"); self.btn_toggle.setCheckable(True)
        self.btn_reset = QPushButton("Reset Plots")
        btn_row.addWidget(self.btn_update)
        btn_row.addWidget(self.btn_r0)
        btn_row.addWidget(self.btn_toggle)
        btn_row.addWidget(self.btn_reset)
        layout.addLayout(btn_row)

        # --- Status labels ---
        self.lab_r0 = QLabel("R₀: -- Ω")
        self.lab_temp = QLabel("Filtered Temp: -- °C")
        self.lab_iv = QLabel("I_meas: -- A | V_meas: -- V")
        for lab in (self.lab_r0, self.lab_temp, self.lab_iv):
            lab.setStyleSheet("font-weight: bold")
            layout.addWidget(lab)

        # --- Plots ---
        self.plot_i = pg.PlotWidget(title="Sourced Current (A)")
        self.plot_i.setBackground('w')
        self.curve_i = self.plot_i.plot([], [], pen=pg.mkPen('black', width=2))
        layout.addWidget(self.plot_i)

        self.plot_t = pg.PlotWidget(title="Temperature (°C)")
        self.plot_t.setBackground('w')
        self.curve_t = self.plot_t.plot([], [], pen=pg.mkPen('blue', width=2))
        self.line_target = pg.InfiniteLine(angle=0, pen=pg.mkPen('black', style=Qt.DashLine))
        self.line_upper = pg.InfiniteLine(angle=0, pen=pg.mkPen('red'))
        self.line_lower = pg.InfiniteLine(angle=0, pen=pg.mkPen('red'))
        self.plot_t.addItem(self.line_target); self.plot_t.addItem(self.line_upper); self.plot_t.addItem(self.line_lower)
        layout.addWidget(self.plot_t)

        # Finalize
        self.setLayout(layout)

        # GUI update timer (reads worker data and runs PID @ 10 Hz)
        self.gui_timer = QTimer(self)
        self.gui_timer.setInterval(100)
        self.gui_timer.timeout.connect(self._on_gui_tick)

    def _wire_runtime(self):
        self.btn_update.clicked.connect(self._apply_smu_settings)
        self.btn_r0.clicked.connect(self._measure_r0)
        self.btn_toggle.clicked.connect(self._toggle_output)
        self.btn_reset.clicked.connect(self._reset_plots)
        # keep shadow setpoint in sync with the UI (block loops handled in setter)
        self.current_setpoint.textEdited.connect(self._on_setpoint_edited)

    # ---------------- Thread-safe setpoint helpers ----------------

    def _set_setpoint(self, value: float):
        """Update both the shadow setpoint and the UI box (signals blocked)."""
        max_i = safe_float(self.max_current, self._default_max_current)
        value = float(np.clip(value, 0.0, max_i))
        self._setpoint_value = value
        self.current_setpoint.blockSignals(True)
        self.current_setpoint.setText(f"{value:.6f}")
        self.current_setpoint.blockSignals(False)

    def _on_setpoint_edited(self, text: str):
        """When user edits the box, keep the shadow setpoint in sync."""
        try:
            val = float(text)
        except Exception:
            return
        max_i = safe_float(self.max_current, self._default_max_current)
        self._setpoint_value = float(np.clip(val, 0.0, max_i))

    # ---------------- SMU settings ----------------

    def _apply_smu_settings(self):
        if not self.smu:
            return
        try:
            vprot = safe_float(self.volt_compliance, 2.0)
            self.smu.write(":SOUR:FUNC CURR")
            self.smu.write(f":SOUR:CURR:RANG {float(self.src_range.currentText())}")
            self.smu.write(f":SENS:CURR:RANG {float(self.meas_i_range.currentText())}")
            self.smu.write(f":SENS:VOLT:RANG {float(self.meas_v_range.currentText())}")
            self.smu.write(f":SENS:VOLT:PROT {vprot}")
            self.smu.write(":SENS:VOLT:APER 0.01")
            self.smu.write(":SENS:CURR:APER 0.01")
            self.smu.write(":SENS:CURR:PROT 1")
            self.smu.write("SENS:REM ON")
            print("SMU settings applied.")
        except Exception as e:
            print("Failed to apply SMU settings:", e)

    # ---------------- R0 measurement ----------------

    def _measure_r0(self):
        if not self.smu:
            return
        try:
            i_r0 = safe_float(self.r0_current, 0.005)
            n = int(safe_float(self.r0_samples, 100))
            dly = max(safe_float(self.r0_delay_s, 0.01), 0.002)

            print(f"Measuring R0 at {i_r0} A, {n} samples, delay {dly}s")
            self.smu.write("OUTP ON")
            self.smu.write(f"SOUR:CURR {i_r0:.6f}")

            vals = []
            for _ in range(n):
                # Always use measured current for resistance
                self.smu.write("MEAS:CURR?")
                i_meas = float(self.smu.read())
                self.smu.write("MEAS:VOLT?")
                v_meas = float(self.smu.read())
                if isfinite(i_meas) and i_meas > 0:
                    vals.append(v_meas / i_meas)
                time.sleep(dly)

            if vals:
                r0 = float(np.mean(vals))
                if isfinite(r0) and r0 > 0:
                    self.r0_measured = r0
                    self.lab_r0.setText(f"R₀: {r0:.4f} Ω")
                    print(f"R0 measured: {r0:.6f} Ω")
                else:
                    print("Invalid R0 result.")
                    self.r0_measured = None
            else:
                print("No valid R0 samples.")
                self.r0_measured = None

        except Exception as e:
            print("R0 measurement failed:", e)

        finally:
            # Critical: turn OFF and reset setpoint so the loop never sticks at i_r0
            try:
                self.smu.write("OUTP OFF")
            except Exception:
                pass
            self._set_setpoint(self._warmup_current)  # warmup start

    # ---------------- Start/stop output ----------------

    def _toggle_output(self):
        if not self.smu:
            return

        if self.btn_toggle.isChecked():
            if self.r0_measured is None:
                print("⚠ Please measure R₀ first.")
                self.btn_toggle.setChecked(False)
                return

            # Reset PID state & start from warmup
            self.pid_int = 0.0
            self.pid_prev_err = 0.0
            self._set_setpoint(self._warmup_current)

            # Worker clamp from UI
            max_i = safe_float(self.max_current, self._default_max_current)
            self.smu.write("OUTP ON")

            # Start worker thread (NOTE: worker reads the shadow float, not the QLineEdit)
            self.worker = SMUWorker(
                smu=self.smu,
                get_target_current_func=lambda: self._setpoint_value,
                max_current_clamp=max_i,
                max_slew_a_per_s=10.0,
            )
            self.thread = QThread()
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.start)
            self.worker.data_ready.connect(self._on_worker_data)  # connect here
            self.worker.finished.connect(self.gui_timer.stop)
            self.worker.finished.connect(lambda: self.smu.write("OUTP OFF"))
            self.thread.start()

            # Start the GUI timer that runs PID & plots
            self.gui_timer.start()
            self.btn_toggle.setText("Stop Output")
            print("Output started.")

        else:
            # Stop
            try:
                if self.worker:
                    self.worker.stop()
                if self.thread:
                    self.thread.quit()
                    self.thread.wait()
                if self.smu:
                    self.smu.write("OUTP OFF")
            except Exception:
                pass
            self.gui_timer.stop()
            self.btn_toggle.setText("Start Output")
            print("Output stopped.")

    # ---------------- GUI tick: PID placeholder ----------------

    def _on_gui_tick(self):
        # PID & plotting happen when worker emits data; timer exists to keep cadence
        pass

    # ---------------- Data handler: PID + plotting ----------------

    def _on_worker_data(self, t, i_meas, v_meas, r_meas):
        # 1) Compute temperature
        t0 = safe_float(self.room_temp, 22.0)
        alpha = safe_float(self.alpha, 0.00381)
        temp = calculate_temperature(r_meas, self.r0_measured, alpha=alpha, t0=t0)

        # 2) Update filter
        self.temp_filter_buf.append(temp)
        if len(self.temp_filter_buf) > self._temp_filter_window:
            self.temp_filter_buf.pop(0)
        valid = [x for x in self.temp_filter_buf if isfinite(x)]
        base_temp = float(np.mean(valid)) if valid else np.nan

        # 3) Update target/limit lines
        tgt = safe_float(self.target_temp, 37.5)
        band = abs(safe_float(self.temp_band, 1.0))
        self.line_target.setPos(tgt)
        self.line_upper.setPos(tgt + band)
        self.line_lower.setPos(tgt - band)

        # 4) PID: warm-up until temp is valid, then regulate
        if not isfinite(base_temp):
            # force warmup current
            self._set_setpoint(self._warmup_current)
        else:
            err = tgt - base_temp
            dt = max(1e-6, self.gui_timer.interval() / 1000.0)

            # Simple gain scheduling
            if abs(err) > 10:
                Kp, Ki, Kd = 0.0010, 0.00010, 0.00050
            elif abs(err) > 5:
                Kp, Ki, Kd = 0.0006, 0.00005, 0.00030
            else:
                Kp, Ki, Kd = 0.0003, 0.00002, 0.00010

            # PID terms with anti-windup and derivative clamp
            self.pid_int = float(np.clip(self.pid_int + err * dt, -self._int_clip, self._int_clip))
            d = np.clip((err - self.pid_prev_err) / dt, -self._deriv_clip, self._deriv_clip)
            adj = Kp * err + Ki * self.pid_int + Kd * d

            # Setpoint update (clamped in setter; worker also clamps and slew-limits)
            i_new = max(0.0, self._setpoint_value + adj)
            self._set_setpoint(i_new)
            self.pid_prev_err = err

        # 5) Buffers & plots
        self.t_buf.append(t); self.i_buf.append(i_meas); self.v_buf.append(v_meas); self.r_buf.append(r_meas); self.temp_buf.append(temp)
        MAX = 2000
        if len(self.t_buf) > MAX:
            self.t_buf = self.t_buf[-MAX:]
            self.i_buf = self.i_buf[-MAX:]
            self.v_buf = self.v_buf[-MAX:]
            self.r_buf = self.r_buf[-MAX:]
            self.temp_buf = self.temp_buf[-MAX:]

        self.curve_i.setData(self.t_buf, self.i_buf)
        self.curve_t.setData(self.t_buf, self.temp_buf)

        # 6) Labels
        self.lab_iv.setText(f"I_meas: {i_meas:.6f} A | V_meas: {v_meas:.6f} V")
        if isfinite(base_temp):
            self.lab_temp.setText(f"Filtered Temp: {base_temp:.2f} °C")
        else:
            self.lab_temp.setText("Filtered Temp: -- °C")

    # ---------------- Misc UI actions ----------------

    def _reset_plots(self):
        self.t_buf.clear(); self.i_buf.clear(); self.v_buf.clear(); self.r_buf.clear(); self.temp_buf.clear()
        self.temp_filter_buf.clear()
        self.curve_i.setData([], [])
        self.curve_t.setData([], [])

    # Ensure we shut down cleanly
    def closeEvent(self, event):
        try:
            if self.worker:
                self.worker.stop()
            if self.thread:
                self.thread.quit()
                self.thread.wait(1000)
            if self.smu:
                self.smu.write("OUTP OFF")
        except Exception:
            pass
        super().closeEvent(event)


# ============ Application ============

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = SMUGUI()
    gui.show()
    sys.exit(app.exec_())
