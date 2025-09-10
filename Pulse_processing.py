import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from sklearn.linear_model import LinearRegression
import os


class PulseAnalyzer:
    def __init__(self, file_path, pulse_time=0.1, sample_rate=800, time_step=1.25, tcr=0.002212):
        self.file_path = file_path
        self.pulse_time = pulse_time
        self.sample_rate = sample_rate
        self.time_step = time_step  # in ms
        self.tcr = tcr

        # Computed values
        self.pulse_data = int(pulse_time * sample_rate)
        self.intervals = []

        # figure handles
        self.fig, self.axs = None, None
        self.preview_fig = None

        # results cache
        self.last_offsets = None
        self.last_slopes = None
        self.last_r2 = None

        # analysis mode ("slope" or "offset")
        self.analysis_mode = "slope"

        # Load data
        self.time_data, self.resistance_data, self.full_temp_data, self.tsensor_data, self.num_pulses, self.time_for_plotting = self._load_data()

    # -----------------------------
    # Data Handling
    # -----------------------------
    def _load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        df_raw = pd.read_csv(
            self.file_path,
            encoding="utf-8",
            delimiter=",",
            header=None,
            skiprows=range(0, 33),
            on_bad_lines="skip",
        )

        data = np.array(df_raw)
        if data.shape[1] < 9:
            raise ValueError("Data does not have enough columns.")

        start_index = np.where(np.diff(data[:, 8] != 0))[0][0] + 1
        time_data = data[start_index:, 0]
        resistance_data = data[start_index:, 3]
        full_temp_data = self.resistance_to_temp(resistance_data, resistance_data[0])
        tsensor_data = data[start_index:, 28].astype(float)

        num_pulses = len(time_data) // self.pulse_data
        time_for_plotting = np.array([time_data[i * self.pulse_data] for i in range(num_pulses)])

        return time_data, resistance_data, full_temp_data, tsensor_data, num_pulses, time_for_plotting

    def resistance_to_temp(self, res_vector, R0):
        return ((res_vector / R0) - 1) / self.tcr

    # -----------------------------
    # Core Analysis
    # -----------------------------
    def fit_sqrt_linear(self, t_seconds, temp, start_ms, end_ms):
        mask = (t_seconds >= start_ms/1000.0) & (t_seconds <= end_ms/1000.0)
        if np.count_nonzero(mask) < 3:
            return np.nan, np.nan, np.nan
        t_slice = t_seconds[mask]
        sqrt_t = np.sqrt(t_slice).reshape(-1, 1)
        temp_slice = temp[mask]
        model = LinearRegression().fit(sqrt_t, temp_slice)
        r2 = model.score(sqrt_t, temp_slice)
        return float(model.coef_[0]), float(model.intercept_), float(r2)

    def compute_offsets_over_pulses(self, start_ms=3000.0, end_ms=4000.0):
        t_seconds = np.arange(self.pulse_data) / float(self.sample_rate)
        slopes = np.full(self.num_pulses, np.nan)
        intercepts = np.full(self.num_pulses, np.nan)
        r2s = np.full(self.num_pulses, np.nan)

        for i in range(self.num_pulses):
            seg = self.resistance_data[i * self.pulse_data : (i + 1) * self.pulse_data]
            if len(seg) < len(t_seconds):
                continue
            temp = self.resistance_to_temp(seg, seg[0])
            s, b, r2 = self.fit_sqrt_linear(t_seconds, temp, start_ms, end_ms)
            slopes[i] = s
            intercepts[i] = b
            r2s[i] = r2

        self.last_slopes = slopes
        self.last_offsets = intercepts
        self.last_r2 = r2s
        return slopes, intercepts, r2s

    # -----------------------------
    # Visualization
    # -----------------------------
    def preview_pulses(self, n_preview=6, return_fig=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        self._plot_preview(ax, n_preview)
        fig.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
            return None

    def _plot_preview(self, ax, n_preview=6):
        ax.cla()
        time_axis_ms = np.arange(self.pulse_data) * (1 / self.sample_rate) * 1000
        preview_indices = np.linspace(0, max(0, self.num_pulses - 1), n_preview, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, len(preview_indices)))

        for i, idx in enumerate(preview_indices):
            start = idx * self.pulse_data
            segment = self.resistance_data[start : start + self.pulse_data]
            if len(segment) == 0:
                continue
            temp_segment = self.resistance_to_temp(segment, segment[0])
            ax.plot(time_axis_ms, temp_segment, label=f"{self.time_for_plotting[idx]:.2f} s", color=colors[i])

        ax.set_title("Overlapped Pulses at Different Times")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        ax.grid(True)

    def setup_main_plot(self):
        self.fig, self.axs = plt.subplots(3, 2, figsize=(12, 12))  # Changed to 3x2 grid
        self.axs = self.axs.flatten()
        plt.subplots_adjust(right=0.75, hspace=0.4, wspace=0.3)

    def update_plot(self):
        if self.axs is None:
            raise RuntimeError("Call setup_main_plot() before update_plot().")
    
        for ax in self.axs:
            ax.cla()
    
        colorset = plt.cm.tab10(np.linspace(0, 1, 10))
    
        for idx, (start_ms, end_ms) in enumerate(self.intervals):
            slopes, intercepts, r2s = self.compute_offsets_over_pulses(start_ms, end_ms)
            label = f"{start_ms:.1f}-{end_ms:.1f} ms"
    
            if self.analysis_mode == "slope":
                self.axs[0].scatter(self.time_for_plotting, slopes, label=label,
                                    color=colorset[idx % 10], s=10)
                init = slopes[~np.isnan(slopes)][0] if np.any(~np.isnan(slopes)) else np.nan
                percent_change = (slopes - init) / init * 100 if not np.isnan(init) else np.full_like(slopes, np.nan)
                self.axs[1].scatter(self.time_for_plotting, percent_change, label=label,
                                    color=colorset[idx % 10], s=10)
                self.axs[2].scatter(self.time_for_plotting, r2s, label=label,
                                    color=colorset[idx % 10], s=10)
    
            elif self.analysis_mode == "offset":
                self.axs[0].scatter(self.time_for_plotting, intercepts, label=label,
                                    color=colorset[idx % 10], s=10)
                init = intercepts[~np.isnan(intercepts)][0] if np.any(~np.isnan(intercepts)) else np.nan
                percent_change = (intercepts - init) / init * 100 if not np.isnan(init) else np.full_like(intercepts, np.nan)
                self.axs[1].scatter(self.time_for_plotting, percent_change, label=label,
                                    color=colorset[idx % 10], s=10)
                self.axs[2].scatter(self.time_for_plotting, r2s, label=label,
                                    color=colorset[idx % 10], s=10)
    
        # --- 4th plot: smoothed temperature trend ---
        smoothed_temp = np.convolve(
            self.tsensor_data[: self.num_pulses * self.pulse_data : self.pulse_data],
            np.ones(10) / 10,
            mode="same",
        )
        self.axs[3].scatter(self.time_for_plotting, smoothed_temp, s=10)
    
        # --- 5th plot: raw temperature ---
        self.axs[4].plot(self.time_data, self.tsensor_data, linewidth=1)
    
        # --- 6th plot: first data point of each pulse ---
        first_points = self.tsensor_data[::self.pulse_data][:self.num_pulses]
        self.axs[5].scatter(self.time_for_plotting, first_points, s=10)
    
        # --- Titles and labels ---
        if self.analysis_mode == "slope":
            titles = ["Slope Over Time", "Δ Slope (%)", "R²", "Smoothed Temperature Trend", "Raw Temperature", "First Pulse Temperature"]
            ylabels = ["Slope", "Change (%)", "R²", "Temp (°C)", "Temp (°C)", "Temp (°C)"]
        else:
            titles = ["Offset Over Time", "Δ Offset (%)", "R²", "Smoothed Temperature Trend", "Raw Temperature", "First Pulse Temperature"]
            ylabels = ["Offset", "Change (%)", "R²", "Temp (°C)", "Temp (°C)", "Temp (°C)"]
    
        for ax, title, ylabel in zip(self.axs, titles, ylabels):
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Time (s)")
            ax.grid(True)
    
        if len(self.intervals) > 0:
            self.axs[0].legend()
            self.axs[1].legend()
    
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            plt.draw()

    # -----------------------------
    # UI Controls 
    # -----------------------------
    def setup_controls(self):
        plt.figure(self.fig.number)
    
        # Make space for sidebar
        plt.subplots_adjust(right=0.75)
    
        # Positions are [left, bottom, width, height] in figure coords
        # Sidebar starts at x=0.85
        x0, w, h = 0.85, 0.13, 0.05
        y = 0.8
    
        # --- Fit controls ---
        ax_start_text = plt.axes([x0, y, w, h]); y -= 0.07
        ax_end_text   = plt.axes([x0, y, w, h]); y -= 0.07
        ax_time_step  = plt.axes([x0, y, w, h]); y -= 0.10
    
        self.txt_start = TextBox(ax_start_text, 'Fit Start (ms)')
        self.txt_end   = TextBox(ax_end_text,   'Fit End (ms)')
        self.txt_time_step = TextBox(ax_time_step, 'Time Step (ms)',  initial=f"{self.time_step}")
    
        # --- Interval controls ---
        ax_button_add    = plt.axes([x0, y, w, h]); y -= 0.07
        ax_button_remove = plt.axes([x0, y, w, h]); y -= 0.10
    
        self.btn_add = Button(ax_button_add, 'Add Interval', color='lightblue', hovercolor='skyblue')
        self.btn_remove = Button(ax_button_remove, 'Remove Interval', color='lightcoral', hovercolor='salmon')
    
        # --- Processing toggle ---
        ax_button_toggle = plt.axes([x0, y, w, h]); y -= 0.10
        self.btn_toggle = Button(ax_button_toggle, 'Mode: Slope', color='lightgray', hovercolor='gray')
    
        # --- Export button only ---
        ax_button_export  = plt.axes([x0, y, w, h]); y -= 0.07
        self.btn_export  = Button(ax_button_export, 'Export CSV', color='lightgray', hovercolor='gray')
    
        # Connect callbacks
        self.btn_add.on_clicked(self._on_add_button_clicked)
        self.btn_remove.on_clicked(self._on_remove_button_clicked)
        self.btn_toggle.on_clicked(self._on_toggle_mode_clicked)
        self.btn_export.on_clicked(self._on_export_clicked)

    def _on_add_button_clicked(self, event):
        try:
            start_val = float(self.txt_start.text)
            end_val = float(self.txt_end.text)
            if start_val < end_val and (start_val, end_val) not in self.intervals:
                self.intervals.append((start_val, end_val))
                self.update_plot()
        except ValueError:
            pass

    def _on_remove_button_clicked(self, event):
        try:
            start_val = float(self.txt_start.text)
            end_val = float(self.txt_end.text)
            if (start_val, end_val) in self.intervals:
                self.intervals.remove((start_val, end_val))
                self.update_plot()
        except ValueError:
            pass

    def _on_update_time_step_clicked(self, event):
        try:
            new_ts = float(self.txt_time_step.text)
            self.time_step = new_ts
            self.update_plot()
            if self.preview_fig is not None:
                ax = self.preview_fig.axes[0]
                self._plot_preview(ax)
                self.preview_fig.canvas.draw_idle()
        except ValueError:
            pass

    def _on_toggle_mode_clicked(self, event):
        # Switch mode
        self.analysis_mode = "offset" if self.analysis_mode == "slope" else "slope"
        print(f"Switched to {self.analysis_mode} mode")
    
        # Clear previous results so graphs reset
        self.last_slopes = None
        self.last_offsets = None
        self.last_r2 = None
    
        # Optionally clear intervals too (uncomment if you want this behavior)
        # self.intervals.clear()
    
        # Redraw empty plots
        self.update_plot()
    
    def _on_export_clicked(self, event):
        if self.last_offsets is None:
            return
        df = pd.DataFrame({
            'time_s': self.time_for_plotting,
            'slope': self.last_slopes,
            'offset': self.last_offsets,
            'r2': self.last_r2,
        })
        out = os.path.join(os.getcwd(), 'tto_results.csv')
        df.to_csv(out, index=False)
        print(f"Exported results to: {out}")

    # -----------------------------
    # Runner
    # -----------------------------
    def run(self):
        self.preview_fig = self.preview_pulses(return_fig=True)
        self.setup_main_plot()
        self.setup_controls()
        self.update_plot()
        plt.show()


if __name__ == "__main__":
    dataset_file = r"C:\Users\lucp12726\Documents\Heater pulsing condensates\250910_Heater_htm\250910_13h23m09s_small_and_big_heater_both_htm.csv"
    analyzer = PulseAnalyzer(dataset_file)
    analyzer.run()