import numpy as np
import pyvisa
import sys
import time
import math
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QLineEdit, QLabel, QHBoxLayout, QGridLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import pyqtgraph as pg


class SMUWorker(QObject):
    data_ready = pyqtSignal(float, float, float, float)  # t, current, voltage, resistance
    finished = pyqtSignal()

    def __init__(self, smu, get_params_func):
        super().__init__()
        self.smu = smu
        self.get_params = get_params_func
        self.running = False
        self.latest_data = None
        self.start_time = time.time()

        
    def start(self):
        self.running = True
        while self.running:
            try:
                t = time.time() - self.start_time
                
                amp, freq, offset = self.get_params()
                current = amp * math.sin(2 * math.pi * freq * t) + offset
                current = max(0.0, min(current, 0.52))  # Clamp to 0–0.4 A

                self.smu.write(f"SOUR:CURR {current:.5f}")

                self.smu.write("MEAS:VOLT?")
                voltage = float(self.smu.read())

                resistance = voltage / current if current > 0 else 0

                self.latest_data = (time.time() - self.start_time, current, voltage, resistance)


            except Exception as e:
                print("Worker error:", e)
                self.running = False

        self.finished.emit()

    def stop(self):
        self.running = False

def calculate_temperature(r, r0, alpha=0.00381, t0=22.0):
        if np.isnan(r) or r0 is None or np.isnan(r0) or r0 == 0:
            return np.nan
        return (r / r0 - 1.0) / alpha + t0

class SMUGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMU Current Control")
        self.resize(1200, 600)

        self.smu = self.initialize_smu()

        self.currents = []
        self.voltages = []
        self.resistances = []
        self.time_stamps = []
        self.temperatures = []
        self.error_sum = 0.0
        self.heating_phase = "warmup"
        self.dynamic_offset = 0.05
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.filtered_temps = []
        self.stable_time = 0.0
        self.temp_osc_target = 1.0  # degrees C




        self.init_ui()

    def initialize_smu(self):
        rm = pyvisa.ResourceManager()
        for addr in rm.list_resources():
            try:
                inst = rm.open_resource(addr)
                idn = inst.query("*IDN?")
                if "KEYSIGHT" in idn.upper() and "B2901A" in idn.upper():
                    print(f"Connected to: {idn.strip()}")

                    inst.timeout = 5000
                    inst.write_termination = '\n'
                    inst.read_termination = '\n'

                    inst.write("*RST")
                    inst.write("*CLS")
                    inst.write(":SOUR:FUNC:MODE CURR")
                    inst.write(":SENS:CURR:PROT 1")
                    inst.write("SOUR:CURR 0.0001")
                    inst.write("SOUR:CURR:RANG 0.3")
                    inst.write("SENS:VOLT:PROT 2")
                    inst.write("SENS:VOLT:RANG 2")
                    inst.write("SENS:REM ON")
                    inst.write(":SENS:VOLT:APER 0.01")
                    inst.write("SENS:CURR:RANG 0.4")
                    inst.write(":SENS:CURR:APER 0.01")
                    
                    self.r0_measured = None
                    
                    return inst
            except Exception as e:
                print(f"Could not connect to {addr}: {e}")
        return None

    def reset_data(self):
        # Clear data
        self.time_stamps.clear()
        self.currents.clear()
        self.voltages.clear()
        self.resistances.clear()
        self.temperatures.clear()
        self.heating_phase = "warmup"
        self.dynamic_offset = 0.05
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.filtered_temps.clear()
        self.stable_time = 0.0
        self.temp_osc_target = 1.0  # degrees C

        # Clear plots
        self.curve_current.setData([], [])
        self.curve_temperature.setData([], [])

        self.r0_label.setText("Initial R₀: -- Ω")

    def init_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Room Temperature (°C)"))
        self.room_temp_input = QLineEdit("22.0")
        layout.addWidget(self.room_temp_input)

        def make_control_row(label_text, line_edit, step, min_val=None, max_val=None):
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            row.addWidget(line_edit)

            btn_dec = QPushButton("–")
            btn_inc = QPushButton("+")
            row.addWidget(btn_dec)
            row.addWidget(btn_inc)

            def update(delta):
                try:
                    val = float(line_edit.text())
                    val += delta
                    if min_val is not None:
                        val = max(min_val, val)
                    if max_val is not None:
                        val = min(max_val, val)
                    line_edit.setText(f"{val:.4f}")
                except ValueError:
                    pass

            btn_dec.clicked.connect(lambda: update(-step))
            btn_inc.clicked.connect(lambda: update(step))

            return row

        # === Amplitude Row ===
        self.amp_input = QLineEdit("0.1")
        layout.addLayout(make_control_row("Amplitude (A)", self.amp_input, step=0.01, min_val=0, max_val=0.4))

        # === Frequency Row ===
        self.freq_input = QLineEdit("1.0")
        layout.addLayout(make_control_row("Frequency (Hz)", self.freq_input, step=0.1, min_val=0.01))
        
        self.target_temp_input = QLineEdit("37.5")
        layout.addLayout(make_control_row("Target Temperature (°C)", self.target_temp_input, step=0.1, min_val=0.01))
        
        # === Boundary Adjust Controls ===
        self.boundary_delta = QLineEdit("1")  # initial delta for ±1°C
        layout.addLayout(make_control_row("Boundary Δ (°C)", self.boundary_delta, step=0.1, min_val=0.01))

        # === Labels and Buttons ===
        self.r0_label = QLabel("Initial R₀: -- Ω")
        layout.addWidget(self.r0_label)
        self.temp_label = QLabel("Measured Temp: -- °C")
        layout.addWidget(self.temp_label)


        
        self.toggle_btn = QPushButton("Start Output")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.toggle_output)
        layout.addWidget(self.toggle_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_data)
        layout.addWidget(self.reset_btn)

        # Current Plot (Blue)
        self.plot_current = pg.PlotWidget(title="Sourced Current (A)")
        self.plot_current.setBackground('w')
        self.curve_current = self.plot_current.plot([], [], pen=pg.mkPen('black', width=2))
        layout.addWidget(self.plot_current)


        # Temperature Plot (Magenta on white background)
        self.plot_temperature = pg.PlotWidget(title="Temperature (°C)")
        self.plot_temperature.setBackground('w')
        self.curve_temperature = self.plot_temperature.plot([], [], pen=pg.mkPen('blue', width=2))

        # Main target line
        self.target_temp_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('black', style=pg.QtCore.Qt.DashLine))
        self.plot_temperature.addItem(self.target_temp_line)

        # Upper bound line (+1°C)
        self.upper_temp_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('red'))
        self.plot_temperature.addItem(self.upper_temp_line)

        # Lower bound line (–1°C)
        self.lower_temp_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('red'))
        self.plot_temperature.addItem(self.lower_temp_line)
        
        layout.addWidget(self.plot_temperature)

        self.setLayout(layout)

    def toggle_output(self):
        if not self.smu:
            return
        print(self.toggle_btn.isChecked())
        if self.toggle_btn.isChecked():
            self.smu.write("OUTP ON")

            try:
                low_current = 0.005  # 1 mA
                num_samples = 100
                resistances = []

                self.smu.write(f"SOUR:CURR {low_current}")

                for _ in range(num_samples):
                    self.smu.write("MEAS:CURR?")
                    i = float(self.smu.read())
                    self.smu.write("MEAS:VOLT?")
                    v = float(self.smu.read())

                    if i > 0:
                        r = v / i
                        resistances.append(r)
                    time.sleep(0.01)

                if resistances:
                    self.r0_measured = sum(resistances) / len(resistances)
                    self.r0_label.setText(f"Initial R₀: {self.r0_measured:.4f} Ω")
                else:
                    self.r0_measured = None
                    print("No valid resistance measurements collected.")

            except Exception as e:
                print("Initial R0 measurement failed:", e)
                self.r0_measured = None

            self.start_time = time.time()
            self.start_thread()
            self.toggle_btn.setText("Stop Output")

        else:
            self.smu.write("OUTP OFF")
            self.stop_thread()
            self.toggle_btn.setText("Start Output")

    def start_thread(self):
        self.worker = SMUWorker(
            smu=self.smu,
            get_params_func=self.get_wave_params
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        
        self.update_timer = pg.QtCore.QTimer()
        self.update_timer.setInterval(16)  # 100 ms update rate
        self.update_timer.timeout.connect(self.handle_data)
        self.update_timer.start()
        
        self.thread.started.connect(self.worker.start)
        self.worker.finished.connect(self.update_timer.stop)

        self.thread.start()

    def stop_thread(self):
        if hasattr(self, 'worker'):
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
    
    def get_wave_params(self):
        try:
            amp = float(self.amp_input.text())
            freq = float(self.freq_input.text())
        except ValueError:
            amp, freq = 0.0, 0.0

        # Only return sine if allowed
        if self.heating_phase == "regulate":
            return amp, freq, self.dynamic_offset
        else:
            return 0.0, 0.0, self.dynamic_offset  # No sine wave yet
 
    def handle_data(self):
        if not hasattr(self.worker, "latest_data") or self.worker.latest_data is None:
            return

        t, current, voltage, resistance = self.worker.latest_data 

        # Calculate temperature
        if self.r0_measured:
            try:
                t0 = float(self.room_temp_input.text())
            except ValueError:
                t0 = 22.0
            temp = calculate_temperature(resistance, self.r0_measured, alpha=0.00381, t0=t0)
        else:
            temp = np.nan

        # Target temp
        try:
            target_temp = float(self.target_temp_input.text())
            self.target_temp_line.setPos(target_temp)
            self.upper_temp_line.setPos(target_temp + float(self.boundary_delta.text()))
            self.lower_temp_line.setPos(target_temp - float(self.boundary_delta.text()))

        except ValueError:
            target_temp = 50.0

        # Filter temp
        self.filtered_temps.append(temp)
        if len(self.filtered_temps) > 100:
            self.filtered_temps.pop(0)
        base_temp = np.mean(self.filtered_temps)

        error = target_temp - base_temp
        dt = 0.016  # ~60Hz update rate


        if self.heating_phase == "warmup":
            self._apply_pid(error, dt)

            if abs(error) <= 0.5:
                self.stable_time += dt
            else:
                self.stable_time = 0.0

            if self.stable_time >= 6.0:
                print("Entering regulation (sine) phase...")
                self.heating_phase = "regulate"

        elif self.heating_phase == "regulate":
            self._apply_pid(error, dt)
            
        if self.heating_phase == "regulate":
            self._apply_pid(error, dt)

        self.temp_label.setText(f"Offset Temp: {base_temp:.2f} °C")

        self._update_plots_and_store(t, current, voltage, resistance, temp)

    def _apply_pid(self, error, dt):
        # Dynamic gain tuning
        if abs(error) > 7:
            print('A')
            self.Kp, self.Ki, self.Kd = 0.00028, 0.00002, 0.00024
        elif abs(error) > 4.5:
            self.Kp, self.Ki, self.Kd = 0.00028, 0.00002, 0.00024
            print('B')
        elif abs(error) > 0.5:
            self.Kp, self.Ki, self.Kd = 0.0001, 0.000, 0.0001
            print('C')

        # PID terms
        self.pid_integral += error * dt
        max_derivative = 30  # or some appropriate value
        derivative = np.clip((error - self.pid_prev_error) / dt, -max_derivative, max_derivative)

        adjustment = self.Kp * error + self.Ki * self.pid_integral + self.Kd * derivative

        self.dynamic_offset += adjustment
        self.dynamic_offset = max(0.001, min(self.dynamic_offset, 0.5))

        self.pid_prev_error = error
        
    def _update_plots_and_store(self, t, current, voltage, resistance, temp):
        self.temperatures.append(temp)
        self.time_stamps.append(t)
        self.currents.append(current)
        self.voltages.append(voltage)
        self.resistances.append(resistance)

        MAX_POINTS = 1000
        self.time_stamps = self.time_stamps[-MAX_POINTS:]
        self.currents = self.currents[-MAX_POINTS:]
        self.voltages = self.voltages[-MAX_POINTS:]
        self.resistances = self.resistances[-MAX_POINTS:]
        self.temperatures = self.temperatures[-MAX_POINTS:]

        self.curve_temperature.setData(self.time_stamps, self.temperatures)
        self.curve_current.setData(self.time_stamps, self.currents)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = SMUGUI()
    gui.show()
    sys.exit(app.exec_())