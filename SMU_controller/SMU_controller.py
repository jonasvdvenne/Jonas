# qt_app.py
import sys, datetime
from pathlib import Path
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
from smu.smu_device import KeysightB2901, MockSMU
from smu.control import Controller
from smu.config import list_profiles, load_profile, save_profile, PIDProfile, PIDBand
from smu.data_io import save_csv, save_json
from smu.utils import DEFAULT_ALPHA, DEFAULT_ROOM_TEMP_C
from smu.method import list_methods, load_method, save_method, MethodRunner, Method, MethodStep


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keysight B2901 Controller (PyQt)")
        self.resize(1400, 900)
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.ctl: Controller | None = None
    
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
    
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(splitter)
    
        # Left tabs
        self.controls = QtWidgets.QTabWidget()
        splitter.addWidget(self.controls)
    
        # --- Tabs ---
        self._init_connection_tab()
        self._init_params_tab()
        self._init_r0_tab()
        self._init_pulse_tab()
        self._init_run_tab()
        self._init_status_tab()
        self._init_method_tab()
        self._init_export_tab()
        self._init_profile_tab()
    
        # Right side container (plots + temp box stacked vertically)
        right_side = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_side)
        splitter.addWidget(right_side)
    
        # Plot widget
        self.plot_widget = pg.GraphicsLayoutWidget()
        right_layout.addWidget(self.plot_widget)
    
        # Current plot
        self.curve_i = self.plot_widget.addPlot(row=0, col=0, title="Current (A)").plot(
            pen="y", symbol="o", symbolSize=4, symbolBrush="y"
        )
        # Voltage plot
        self.curve_v = self.plot_widget.addPlot(row=1, col=0, title="Voltage (V)").plot(
            pen="c", symbol="o", symbolSize=4, symbolBrush="c"
        )
        # Temperature plot
        self.curve_t = self.plot_widget.addPlot(row=2, col=0, title="Temperature (°C)").plot(
            pen="m", symbol="o", symbolSize=4, symbolBrush="m"
        )
        # Slopes plot
        self.curve_slope = self.plot_widget.addPlot(row=3, col=0, title="Slopes (°C/s)").plot(
            pen="g", symbol="t", symbolSize=6, symbolBrush="g"
        )
    
        # Big temperature display below plots
        self.temp_display = QtWidgets.QLabel("— °C")
        self.temp_display.setAlignment(QtCore.Qt.AlignCenter)
        self.temp_display.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: darkred;
            border: 2px solid gray;
            padding: 10px;
        """)
        right_layout.addWidget(self.temp_display)
    
        # Adjust splitter sizes
        splitter.setSizes([400, 1000])
    
        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(50)




    # ---------------- Tabs ----------------
    def _init_connection_tab(self):
        w = QtWidgets.QWidget(); l = QtWidgets.QFormLayout(w)
        self.use_mock = QtWidgets.QCheckBox("Use Mock SMU")
        self.resource_edit = QtWidgets.QLineEdit()
        self.profile_combo = QtWidgets.QComboBox()
        for nm in list_profiles().keys():
            self.profile_combo.addItem(nm)
        self.connect_btn = QtWidgets.QPushButton("Connect / Initialize")
        l.addRow(self.use_mock); l.addRow("Resource:", self.resource_edit)
        l.addRow("Profile:", self.profile_combo); l.addRow(self.connect_btn)
        self.connect_btn.clicked.connect(self.connect_controller)
        self.controls.addTab(w, "Connection")

    def _init_params_tab(self):
        w = QtWidgets.QWidget(); l = QtWidgets.QFormLayout(w)
    
        self.room_temp = QtWidgets.QDoubleSpinBox()
        self.room_temp.setValue(DEFAULT_ROOM_TEMP_C)
    
        self.band = QtWidgets.QDoubleSpinBox(); self.band.setValue(1.0)
        self.i_max = QtWidgets.QDoubleSpinBox(); self.i_max.setValue(0.5)
        self.v_comp = QtWidgets.QDoubleSpinBox(); self.v_comp.setValue(2.0)
    
        # Range selectors
        self.curr_range = QtWidgets.QComboBox()
        self.curr_range.addItems(["Auto", "10 µA", "100 µA", "1 mA", "10 mA", "100 mA", "1 A"])
        self.volt_range = QtWidgets.QComboBox()
        self.volt_range.addItems(["Auto", "20 mV", "200 mV", "2 V", "20 V", "200 V"])
    
        self.apply_params_btn = QtWidgets.QPushButton("Apply Params")
    
        for label, wdg in [
            ("Room Temp (°C)", self.room_temp),
            ("Temp Band (°C)", self.band),
            ("Max Current (A)", self.i_max),
            ("Compliance V", self.v_comp),
            ("Current Range", self.curr_range),
            ("Voltage Range", self.volt_range),
        ]:
            l.addRow(label, wdg)
    
        l.addRow(self.apply_params_btn)
        self.apply_params_btn.clicked.connect(self.apply_params)
        self.controls.addTab(w, "Parameters")



    def _init_r0_tab(self):
        w = QtWidgets.QWidget(); l = QtWidgets.QFormLayout(w)
        self.r0_i = QtWidgets.QDoubleSpinBox(); self.r0_i.setDecimals(6)
        self.r0_d = QtWidgets.QDoubleSpinBox(); self.r0_d.setDecimals(6)
        self.r0_n = QtWidgets.QSpinBox(); self.r0_n.setValue(100)
        self.measure_r0_btn = QtWidgets.QPushButton("Measure R₀")
        self.r0_result = QtWidgets.QLabel("—")
        l.addRow("Current (A)", self.r0_i); l.addRow("Delay (s)", self.r0_d)
        l.addRow("Samples", self.r0_n); l.addRow(self.measure_r0_btn)
        l.addRow("Result", self.r0_result)
        self.measure_r0_btn.clicked.connect(self.measure_r0)
        self.controls.addTab(w, "R₀ Measurement")
        
    def _init_method_tab(self):
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(w)
    
        form = QtWidgets.QFormLayout()
        # --- Method selection ---
        self.method_combo = QtWidgets.QComboBox()
        for nm in list_methods().keys():
            self.method_combo.addItem(nm)
    
        self.start_method_btn = QtWidgets.QPushButton("Start Method")
        self.stop_method_btn = QtWidgets.QPushButton("Stop Method")
        form.addRow("Method:", self.method_combo)
        form.addRow(self.start_method_btn, self.stop_method_btn)
    
        # --- Method creation ---
        self.method_name = QtWidgets.QLineEdit("new_method")
        form.addRow("Name:", self.method_name)
    
        l.addLayout(form)
    
        # Table
        headers = ["Temp (°C)", "Duration (s)", "Pulses", "Power (W)", "Pulse Dur (s)", "Pulse Per (s)"]
        self.method_table = QtWidgets.QTableWidget(3, len(headers))
        self.method_table.setHorizontalHeaderLabels(headers)
    
        # initialize checkboxes in "Pulses" column
        for r in range(self.method_table.rowCount()):
            self._init_pulse_checkbox(r)
    
        l.addWidget(self.method_table)
    
        # Buttons
        btns = QtWidgets.QHBoxLayout()
        self.add_row_btn = QtWidgets.QPushButton("➕ Add Row")
        self.remove_row_btn = QtWidgets.QPushButton("➖ Remove Row")
        self.save_method_btn = QtWidgets.QPushButton("💾 Save Method")
        btns.addWidget(self.add_row_btn)
        btns.addWidget(self.remove_row_btn)
        btns.addWidget(self.save_method_btn)
        l.addLayout(btns)
    
        # --- Connect signals ---
        self.start_method_btn.clicked.connect(self.start_method)
        self.stop_method_btn.clicked.connect(self.stop_method)
        self.save_method_btn.clicked.connect(self.save_method)
        self.add_row_btn.clicked.connect(self.add_method_row)
        self.remove_row_btn.clicked.connect(self.remove_method_row)
    
        self.controls.addTab(w, "Methods")


    def _pulse_checkbox_changed(self, item: QtWidgets.QTableWidgetItem):
        row = item.row()
        col = item.column()
        if col != 2:  # only react to "Pulses" column
            return
    
        enabled = item.checkState() == QtCore.Qt.Checked
        for c in (3, 4, 5):
            cell = self.method_table.item(row, c)
            if not cell:
                cell = QtWidgets.QTableWidgetItem("")
                self.method_table.setItem(row, c, cell)
            if enabled:
                cell.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled)
                cell.setBackground(QtCore.Qt.white)
            else:
                cell.setFlags(QtCore.Qt.ItemIsEnabled)  # read-only
                cell.setText("")
                cell.setBackground(QtCore.Qt.lightGray)
                
    def add_method_row(self):
        r = self.method_table.rowCount()
        self.method_table.insertRow(r)
        self._init_pulse_checkbox(r)
            
    def remove_method_row(self):
        r = self.method_table.currentRow()
        if r >= 0:
            self.method_table.removeRow(r)
         
    def start_method(self):
        if not self.ctl:
            return
        name = self.method_combo.currentText()
        method = load_method(name)
        self.method_runner = MethodRunner(self.ctl, method)
        self.method_runner.start()
        
    def _init_pulse_checkbox(self, row: int):
        chk = QtWidgets.QTableWidgetItem()
        chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        chk.setCheckState(QtCore.Qt.Unchecked)
        self.method_table.setItem(row, 2, chk)
    
        # Disable pulse parameter cells initially
        for col in (3, 4, 5):
            item = QtWidgets.QTableWidgetItem("")
            item.setFlags(QtCore.Qt.ItemIsEnabled)  # not editable until checkbox ticked
            item.setBackground(QtCore.Qt.lightGray)
            self.method_table.setItem(row, col, item)
    
        # Track changes
        self.method_table.itemChanged.connect(self._pulse_checkbox_changed)
      
    def save_method(self):
        steps = []
        for r in range(self.method_table.rowCount()):
            try:
                temp_item = self.method_table.item(r, 0)
                dur_item = self.method_table.item(r, 1)
                pulses_item = self.method_table.item(r, 2)
                pwr_item = self.method_table.item(r, 3)
                pdur_item = self.method_table.item(r, 4)
                pper_item = self.method_table.item(r, 5)
    
                if not (temp_item and dur_item):
                    continue
    
                temp = float(temp_item.text())
                dur = float(dur_item.text())
                pulses = pulses_item and pulses_item.checkState() == QtCore.Qt.Checked
                pwr = float(pwr_item.text()) if pwr_item and pwr_item.text() else None
                pdur = float(pdur_item.text()) if pdur_item and pdur_item.text() else None
                pper = float(pper_item.text()) if pper_item and pper_item.text() else None
    
                steps.append(MethodStep(
                    target_temp_c=temp,
                    duration_s=dur,
                    pulses=pulses,
                    pulse_power_w=pwr,
                    pulse_duration_s=pdur,
                    pulse_period_s=pper
                ))
            except Exception:
                pass
    
        if not steps:
            QtWidgets.QMessageBox.warning(self, "Error", "No valid steps defined.")
            return
    
        method = Method(name=self.method_name.text(), steps=steps)
        save_method(method)
        QtWidgets.QMessageBox.information(self, "Saved", f"Method '{method.name}' saved.")
    
        # refresh combo box
        self.method_combo.clear()
        for nm in list_methods().keys():
            self.method_combo.addItem(nm)



    def stop_method(self):
        if hasattr(self, "method_runner"):
            self.method_runner.stop()

    def _init_pulse_tab(self):
        w = QtWidgets.QWidget(); l = QtWidgets.QFormLayout(w)
        self.pulse_pwr = QtWidgets.QDoubleSpinBox(); self.pulse_pwr.setDecimals(6)
        self.pulse_dur = QtWidgets.QDoubleSpinBox(); self.pulse_dur.setDecimals(6)
        self.pulse_per = QtWidgets.QDoubleSpinBox(); self.pulse_per.setDecimals(6)
        self.apply_pulse_btn = QtWidgets.QPushButton("Apply Pulse")
        l.addRow("Power (W)", self.pulse_pwr)
        l.addRow("Duration (s)", self.pulse_dur)
        l.addRow("Period (s)", self.pulse_per)
        l.addRow(self.apply_pulse_btn)
    
        self.pulse_mode = QtWidgets.QComboBox()
        self.pulse_mode.addItems(["Auto", "Force Paused", "Force Active"])
        l.addRow("Pulse Mode", self.pulse_mode)
    
        # 🔹 Slope fitting window (moved here)
        self.slope_t0 = QtWidgets.QDoubleSpinBox(); self.slope_t0.setDecimals(3)
        self.slope_t1 = QtWidgets.QDoubleSpinBox(); self.slope_t1.setDecimals(3)
        self.slope_t0.setValue(0.0)
        self.slope_t1.setValue(0.5)
        l.addRow("Slope window start (s)", self.slope_t0)
        l.addRow("Slope window end (s)", self.slope_t1)
    
        self.apply_pulse_btn.clicked.connect(self.apply_pulse)
        self.pulse_mode.currentIndexChanged.connect(self.set_pulse_mode)
        self.controls.addTab(w, "Pulses")


    def _init_run_tab(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # Target temperature
        self.target_temp = QtWidgets.QDoubleSpinBox()
        self.target_temp.setValue(37.5)
        self.target_temp.setRange(0, 200)
        v.addWidget(QtWidgets.QLabel("Target Temp (°C)"))
        v.addWidget(self.target_temp)
        self.target_temp.valueChanged.connect(self.set_target_temp)

        # Start/Stop/Reset
        h = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.reset_btn = QtWidgets.QPushButton("Reset Logs")
        for b in [self.start_btn, self.stop_btn, self.reset_btn]:
            h.addWidget(b)
        v.addLayout(h)

        self.start_btn.clicked.connect(lambda: self.ctl.start() if self.ctl else None)
        self.stop_btn.clicked.connect(lambda: self.ctl.stop() if self.ctl else None)
        self.reset_btn.clicked.connect(self.reset_logs)

        self.controls.addTab(w, "Run")

    def _init_export_tab(self):
        w = QtWidgets.QWidget(); l = QtWidgets.QFormLayout(w)
        self.save_dir = QtWidgets.QLineEdit(".")
        self.browse_btn = QtWidgets.QPushButton("Browse…")
        h = QtWidgets.QHBoxLayout(); h.addWidget(self.save_dir); h.addWidget(self.browse_btn)
        cont = QtWidgets.QWidget(); cont.setLayout(h)
        self.base_name = QtWidgets.QLineEdit("run")
        self.save_csv_btn = QtWidgets.QPushButton("Save CSV")
        self.save_json_btn = QtWidgets.QPushButton("Save JSON")
        l.addRow("Save dir", cont); l.addRow("Base name", self.base_name)
        l.addRow(self.save_csv_btn, self.save_json_btn)
        self.browse_btn.clicked.connect(self.pick_dir)
        self.save_csv_btn.clicked.connect(self.save_csv)
        self.save_json_btn.clicked.connect(self.save_json)
        self.controls.addTab(w, "Export")

    def _init_profile_tab(self):
        w = QtWidgets.QWidget(); l = QtWidgets.QFormLayout(w)
        self.prof_name = QtWidgets.QLineEdit("new_profile")
        self.temp_window = QtWidgets.QSpinBox(); self.temp_window.setValue(10)
        self.deriv_clip = QtWidgets.QDoubleSpinBox(); self.deriv_clip.setDecimals(6)
        self.int_clip = QtWidgets.QDoubleSpinBox(); self.int_clip.setDecimals(6)
        self.cooldown = QtWidgets.QDoubleSpinBox(); self.cooldown.setDecimals(6)
        self.drop_limit = QtWidgets.QDoubleSpinBox(); self.drop_limit.setDecimals(6)
    
        # 🔹 New alpha spinbox
        self.prof_alpha = QtWidgets.QDoubleSpinBox()
        self.prof_alpha.setDecimals(6)
        self.prof_alpha.setValue(DEFAULT_ALPHA)
    
        self.bands_table = QtWidgets.QTableWidget(3, 4)
        self.bands_table.setHorizontalHeaderLabels(["|err| >", "Kp", "Ki", "Kd"])
        self.save_prof_btn = QtWidgets.QPushButton("Save Profile")
    
        for lbl, wdg in [
            ("Name", self.prof_name),
            ("Temp filter win", self.temp_window),
            ("Deriv clip", self.deriv_clip),
            ("Int clip", self.int_clip),
            ("Cooldown (s)", self.cooldown),
            ("Drop limit (A)", self.drop_limit),
            ("Alpha", self.prof_alpha),   # 🔹 show alpha here
        ]:
            l.addRow(lbl, wdg)
    
        l.addRow(self.bands_table); l.addRow(self.save_prof_btn)
        self.save_prof_btn.clicked.connect(self.save_profile)
        self.controls.addTab(w, "Profiles")


    def _init_status_tab(self):
        w = QtWidgets.QWidget(); grid = QtWidgets.QGridLayout(w)
        self.status_labels = {}
        fields = ["Running", "Latest Cmd (A)", "Setpoint (A)", "Samples", "Pulse Active", "R₀ (Ω)", "R₀ floor (A)"]
        for i,f in enumerate(fields):
            grid.addWidget(QtWidgets.QLabel(f), i, 0)
            lab = QtWidgets.QLabel("—"); self.status_labels[f] = lab
            grid.addWidget(lab, i, 1)
        self.controls.addTab(w, "Status")

    # ---------------- Actions ----------------
    def connect_controller(self):
        use_mock = self.use_mock.isChecked()
        resource = self.resource_edit.text().strip() or None
        prof_name = self.profile_combo.currentText()
        smu = MockSMU() if use_mock else KeysightB2901(resource)
        prof = load_profile(prof_name)
        self.ctl = Controller(smu=smu, pid_profile=prof)
        self.load_profile_into_ui(prof)
    def apply_params(self):
        if not self.ctl: 
            return
        self.ctl.set_basic_params(
            room_temp_c=self.room_temp.value(),
            target_temp_c=self.ctl.target_temp_c,
            temp_band=self.band.value(),
            max_current=self.i_max.value(),
            volt_compliance=self.v_comp.value(),
            current_range=self.curr_range.currentText(),
            voltage_range=self.volt_range.currentText()
        )


    def apply_pulse(self):
        if not self.ctl: 
            return
        self.ctl.set_pulse(self.pulse_pwr.value(), self.pulse_dur.value(), self.pulse_per.value())
        self.ctl.slope_t0 = self.slope_t0.value()
        self.ctl.slope_t1 = self.slope_t1.value()

    def set_pulse_mode(self, idx):
        if not self.ctl: return
        if idx == 0: self.ctl.pulse_manual_override = None
        elif idx == 1: self.ctl.pulse_manual_override = True
        else: self.ctl.pulse_manual_override = False

    def reset_logs(self):
        if not self.ctl: return
        self.ctl.log_t.clear(); self.ctl.log_i.clear(); self.ctl.log_v.clear()
        self.ctl.log_r.clear(); self.ctl.log_temp.clear(); self.ctl.log_pulse_flag.clear()
        self.ctl.pulse_events.clear()

    def pick_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        if d: self.save_dir.setText(d)

    def save_csv(self):
        if not self.ctl: return
        base = f"{datetime.datetime.now().strftime('%y%m%d')}_{self.base_name.text()}"
        save_csv(str(Path(self.save_dir.text()) / f"{base}.csv"), self.ctl.metadata(), self.ctl.export_rows())

    def save_json(self):
        if not self.ctl: return
        base = f"{datetime.datetime.now().strftime('%y%m%d')}_{self.base_name.text()}"
        save_json(str(Path(self.save_dir.text()) / f"{base}.json"), self.ctl.export_payload())

    def measure_r0(self):
        if not self.ctl:
            return
        r0 = self.ctl.measure_r0(self.r0_i.value(), self.r0_n.value(), self.r0_d.value())
        if r0:
            self.r0_result.setText(f"{r0:.6f} Ω")
            # update status tab immediately
            self.status_labels["R₀ (Ω)"].setText(f"{r0:.6f}")
            self.status_labels["R₀ floor (A)"].setText(f"{self.ctl.r0_floor:.6f}")
        else:
            self.r0_result.setText("Failed")


    def save_profile(self):
        bands = []
        for r in range(self.bands_table.rowCount()):
            try:
                abs_err = float(self.bands_table.item(r,0).text())
                kp = float(self.bands_table.item(r,1).text())
                ki = float(self.bands_table.item(r,2).text())
                kd = float(self.bands_table.item(r,3).text())
                bands.append(PIDBand(abs_error_gt=abs_err, Kp=kp, Ki=ki, Kd=kd))
            except Exception:
                pass
        
        prof = PIDProfile(
            name=self.prof_name.text(),
            temp_filter_window=self.temp_window.value(),
            deriv_clip=self.deriv_clip.value(),
            int_clip=self.int_clip.value(),
            post_pulse_cooldown_s=self.cooldown.value(),
            post_pulse_drop_limit=self.drop_limit.value(),
            alpha=self.prof_alpha.value(),   # 🔹 use new spinbox
            bands=bands
        )

        save_profile(prof)


        save_profile(prof)
    def load_profile_into_ui(self, prof: PIDProfile):
        """Populate the Profiles tab fields from a PIDProfile object."""
        self.prof_name.setText(prof.name)
        self.temp_window.setValue(prof.temp_filter_window)
        self.deriv_clip.setValue(prof.deriv_clip)
        self.int_clip.setValue(prof.int_clip)
        self.cooldown.setValue(prof.post_pulse_cooldown_s)
        self.drop_limit.setValue(prof.post_pulse_drop_limit)
        self.prof_alpha.setValue(prof.alpha)
    
        # update bands table
        self.bands_table.setRowCount(len(prof.bands))
        for r, band in enumerate(prof.bands):
            self.bands_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(band.abs_error_gt)))
            self.bands_table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(band.Kp)))
            self.bands_table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(band.Ki)))
            self.bands_table.setItem(r, 3, QtWidgets.QTableWidgetItem(str(band.Kd)))

    def set_target_temp(self, val):
        if not self.ctl:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Target Temperature",
            f"Apply new target temperature {val:.2f} °C?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.ctl.set_target_temp(val)
            self.status_labels["Setpoint (A)"].setText("Updating…")
        else:
            # reset spinbox to the last value
            self.target_temp.blockSignals(True)
            self.target_temp.setValue(self.ctl.target_temp_c)
            self.target_temp.blockSignals(False)


    def update_status(self):
        if not self.ctl:
            return
    
        t, i, v, _, temp, _ = self.ctl.snapshot()
        if len(t) > 2:
            w = 2000
            self.curve_i.setData(t[-w:], i[-w:])
            self.curve_v.setData(t[-w:], v[-w:])
            self.curve_t.setData(t[-w:], temp[-w:])
    
            # 🔹 Slopes plotting
            slopes = [
                (ev["start_time"], ev["slope_c_per_s"])
                for ev in self.ctl.pulse_events
                if ev.get("slope_computed") and ev.get("slope_c_per_s") is not None
            ]
            if slopes:
                ts, vals = zip(*slopes)
                self.curve_slope.setData(ts, vals)
            else:
                self.curve_slope.clear()
    
        # 🔹 Update temperature display
        status = self.ctl.latest_status()
        last_temp = status.get("last_temp_c", None)
        if last_temp is not None:
            self.temp_display.setText(f"{last_temp:.2f} °C")
        else:
            self.temp_display.setText("— °C")




# ---------------- Main ----------------
if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    mw=MainWindow(); mw.show()
    sys.exit(app.exec_())
