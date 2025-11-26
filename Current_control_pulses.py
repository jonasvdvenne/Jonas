import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvisa
import time


# ==============================================================
# Keysight SMU Class
# ==============================================================

class KeysightSMU:
    def __init__(self, resource_string, timeout=None):
        self.rm = pyvisa.ResourceManager()
        self.device = self.rm.open_resource(resource_string)
        self.device.timeout = timeout
        self.initialize()

    def initialize(self):
        """Reset and initialize the SMU."""
        self.device.write("*RST")
        self.device.write(":SOUR:FUNC:MODE CURR")  # Set to current source mode
        self.device.write(':SENSe1:VOLTage:DC:PROTection:LEVel 3')  # Protection limit
        self.device.write(":SENS:FUNC 'VOLT','CURR'")  # Enable voltage and current measurements
        self.check_error("Initialization")

    def check_error(self, context):
        """Check and print any error from the SMU."""
        error = self.device.query(":SYST:ERR?")
        if error != '0,"No error"':
            print(f"Error in {context}: {error}")

    def configure_current_source(self, current_waveform):
        """Set up the current source waveform."""
        waveform_str = ','.join(map(str, current_waveform))
        self.device.write(":SOUR:CURR:MODE LIST")
        self.device.write(f":SOUR:LIST:CURR {waveform_str}")

    def configure_voltage_measurement(self, voltage_range, aperture_time):
        """Configure voltage measurement settings."""
        self.device.write(":SENS:REM ON")  # Enable 4-wire sensing
        self.device.write(f":SENS:VOLT:APER {aperture_time}")

    def configure_current_measurement(self, aperture_time):
        """Configure current measurement settings."""
        self.device.write(f":SENS:CURR:APER {aperture_time}")

    def set_ranges(self, voltage_range, current_range):
        """Set manual ranges for voltage and current."""
        self.device.write(f":SOUR:CURR:RANG {current_range}")
        self.device.write(f":SENS:VOLT:RANG {voltage_range}")

    def configure_trigger(self, num_points, interval_time):
        """Configure triggering for synchronized measurements."""
        self.device.write(":TRIG:SOUR TIM")  # Set trigger source to time interval
        self.device.write(f":TRIG:COUN {num_points}")  # Number of points to measure
        self.device.write(f":TRIG:TIM {interval_time}")  # Interval between each measurement

    def initiate_measurement(self):
        """Start current sourcing and voltage measurement."""
        self.device.write(":OUTP ON")
        self.device.write(":INIT")

    def fetch_measurements(self):
        """Retrieve measured voltage and current data."""
        voltage_data = self.device.query(":FETC:ARR:VOLT?")
        current_data = self.device.query(":FETC:ARR:CURR?")
        self.check_error("Fetching Measurements")

        voltage_array = np.array(list(map(float, voltage_data.split(','))))
        current_array = np.array(list(map(float, current_data.split(','))))
        
        return voltage_array, current_array

    def close(self):
        """Turn off output and close the connection."""
        self.device.write(":OUTP OFF")
        self.device.close()


# ==============================================================
# Pulse Waveform Generators
# ==============================================================

def generate_pulse_waveform(amplitude, pulse_width, period, num_pulses, offset=0.0, sample_rate=1000):
    """Generate a rectangular pulse waveform."""
    duration = num_pulses * period
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    waveform = np.full_like(t, offset)

    for i in range(num_pulses):
        start_idx = int(i * period * sample_rate)
        end_idx = int((i * period + pulse_width) * sample_rate)
        waveform[start_idx:end_idx] = amplitude + offset

    return waveform, t


def generate_dc_waveform(dc_value, duration, sample_rate):
    """Generate a constant DC waveform."""
    num_samples = int(duration * sample_rate)
    dc_waveform = np.full(num_samples, dc_value)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    return dc_waveform, t


# ==============================================================
# Data Visualization and Saving
# ==============================================================

def plot_results(time, measured_current, measured_voltage, resistance_readings, power_readings, temperature_readings, pulse_state):
    """Plot the measurement data with pulse activity shading."""
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 20))
    labels = ["Current (A)", "Voltage (V)", "Power (W)", "Resistance (Ohms)", "Temperature (°C)"]
    data = [measured_current, measured_voltage, power_readings, resistance_readings, temperature_readings]
    colors = ['blue', 'orange', 'purple', 'green', 'red']

    # --- Shade active pulse regions ---
    in_pulse = False
    start_time = None
    for i in range(len(time)):
        if pulse_state[i] == "Pulse Active" and not in_pulse:
            start_time = time[i]
            in_pulse = True
        elif pulse_state[i] != "Pulse Active" and in_pulse:
            end_time = time[i]
            for ax in axs:
                ax.axvspan(start_time, end_time, color='red', alpha=0.15)
            in_pulse = False

    if in_pulse:
        for ax in axs:
            ax.axvspan(start_time, time[-1], color='red', alpha=0.15)

    # --- Plot data ---
    for ax, label, data_array, color in zip(axs, labels, data, colors):
        ax.scatter(time, data_array, label=label, color=color, s=10)
        ax.set_ylabel(label)
        ax.grid()
        ax.legend()

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def calculate_temperature(resistance):
    """Calculate temperature from resistance using a linear approximation."""
    R0 = 1.82  
    T0 = 22  
    alpha = 0.00381  
    return (resistance / R0 - 1) / alpha + T0


def save_data_to_csv(directory, filename, time, measured_current, measured_voltage, resistance_readings, power_readings, temperature_readings, pulse_state, params):
    """Save measurement data to a CSV file."""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{filename}.csv")

    with open(file_path, 'w') as f:
        f.write("Parameters:\n")
        for key, value in params.items():
            f.write(f"{key},{value}\n")
        f.write("\n")

    data = {
        "Time (s)": time,
        "Measured Current (A)": measured_current,
        "Measured Voltage (V)": measured_voltage,
        "Resistance (Ohms)": resistance_readings,
        "Power (W)": power_readings,
        "Temperature (°C)": temperature_readings,
        "Pulse State": pulse_state,
    }

    df = pd.DataFrame(data)
    df.to_csv(file_path, mode='a', index=False)
    print(f"Data saved to {file_path}")


# ==============================================================
# Measurement Routine (with variable DC offsets)
# ==============================================================

def run_measurement(smu, wave_params, initial_dc_value, initial_dc_duration, inter_dc_values, inter_dc_duration, dc_sample_rate):
    """Run the measurement process and calculate resistance, power, and temperature with variable DC offsets."""
    all_voltage, all_current, all_time = [], [], []
    resistance_readings, power_readings, temperature_readings = [], [], []
    pulse_state = []

    time_offset = 0.0
    current_offset = initial_dc_value  # Start from initial DC offset

    # --- Initial DC step ---
    dc_waveform, dc_time = generate_dc_waveform(current_offset, initial_dc_duration, dc_sample_rate)
    smu.configure_current_source(dc_waveform)
    smu.configure_trigger(len(dc_waveform), 1 / dc_sample_rate)
    smu.initiate_measurement()
    voltage_data, current_data = smu.fetch_measurements()

    all_voltage.extend(voltage_data)
    all_current.extend(current_data)
    all_time.extend(dc_time + time_offset)
    pulse_state.extend(["DC"] * len(dc_waveform))
    time_offset += dc_time[-1]

    for v, c in zip(voltage_data, current_data):
        r = v / c if c != 0 else np.nan
        p = v * c
        t = calculate_temperature(r)
        resistance_readings.append(r)
        power_readings.append(p)
        temperature_readings.append(t)

    # --- Pulse sequences ---
    for i, (amplitude, pulse_width, period, num_pulses, sample_rate) in enumerate(wave_params):
        pulse_wave, pulse_time = generate_pulse_waveform(amplitude, pulse_width, period, num_pulses, offset=current_offset, sample_rate=sample_rate)
        smu.configure_current_source(pulse_wave)
        smu.configure_trigger(len(pulse_wave), 1 / sample_rate)
        smu.initiate_measurement()
        voltage_data, current_data = smu.fetch_measurements()

        all_voltage.extend(voltage_data)
        all_current.extend(current_data)
        all_time.extend(pulse_time + time_offset)

        # Pulse state labeling (active/inactive per sample)
        state_segment = np.full_like(pulse_wave, "Pulse Inactive", dtype=object)
        samples_per_period = int(period * sample_rate)
        samples_active = int(pulse_width * sample_rate)
        for n in range(num_pulses):
            start_idx = n * samples_per_period
            end_idx = start_idx + samples_active
            state_segment[start_idx:end_idx] = "Pulse Active"
        pulse_state.extend(state_segment)

        time_offset += pulse_time[-1]

        for v, c in zip(voltage_data, current_data):
            r = v / c if c != 0 else np.nan
            p = v * c
            t = calculate_temperature(r)
            resistance_readings.append(r)
            power_readings.append(p)
            temperature_readings.append(t)

        # --- Intermediate DC step ---
        if i < len(wave_params) - 1:
            new_offset = inter_dc_values[i] if i < len(inter_dc_values) else inter_dc_values[-1]
            inter_wave, inter_time = generate_dc_waveform(new_offset, inter_dc_duration, dc_sample_rate)
            smu.configure_current_source(inter_wave)
            smu.configure_trigger(len(inter_wave), 1 / dc_sample_rate)
            smu.initiate_measurement()
            voltage_data, current_data = smu.fetch_measurements()

            all_voltage.extend(voltage_data)
            all_current.extend(current_data)
            all_time.extend(inter_time + time_offset)
            pulse_state.extend(["DC"] * len(inter_wave))
            time_offset += inter_time[-1]

            for v, c in zip(voltage_data, current_data):
                r = v / c if c != 0 else np.nan
                p = v * c
                t = calculate_temperature(r)
                resistance_readings.append(r)
                power_readings.append(p)
                temperature_readings.append(t)

            # Update baseline offset for next pulse block
            current_offset = new_offset

    return all_time, all_voltage, all_current, resistance_readings, power_readings, temperature_readings, pulse_state


# ==============================================================
# Main Routine
# ==============================================================

def main():
    # Pulse configuration: (amplitude [A], pulse_width [s], period [s], num_pulses, sample_rate [Hz])
    wave_params = [
        (0.05, 1, 5, 10, 200),
        (0.05, 1, 5, 10, 200),
        (0.05, 1, 5, 10, 200),
        (0.05, 1, 5, 10, 200),
        (0.05, 1, 5, 10, 200),
        (0.05, 1, 5, 10, 200),
        (0.05, 1, 5, 10, 200)
    ]

    # Each inter-DC value defines the offset after each pulse group
    inter_dc_values = [0.075, 0.1, 0.125, 0.15, 0.175, 0.2]  # two inter-DC steps for three pulse groups

    initial_dc_value = 0.05
    initial_dc_duration = 5
    inter_dc_duration = 10
    dc_sample_rate = 50

    smu = KeysightSMU("USB0::0x0957::0x8B18::MY51142506::INSTR")
    smu.set_ranges(voltage_range=2, current_range=1)
    smu.configure_voltage_measurement(voltage_range=2, aperture_time=0.001)
    smu.configure_current_measurement(aperture_time=0.001)

    all_time, all_voltage, all_current, resistance, power, temperature, pulse_state = run_measurement(
        smu, wave_params, initial_dc_value, initial_dc_duration, inter_dc_values, inter_dc_duration, dc_sample_rate
    )

    params = {
        "Initial DC Value (A)": initial_dc_value,
        "Initial DC Duration (s)": initial_dc_duration,
        "Inter-DC Values (A)": inter_dc_values,
        "Inter-DC Duration (s)": inter_dc_duration,
        "Voltage Range (V)": 2,
        "Current Range (A)": 1,
        "Pulse Parameters": str(wave_params)
    }

    plot_results(np.array(all_time), np.array(all_current), np.array(all_voltage), resistance, power, temperature, pulse_state)

    if input("Do you want to save the measurement data to a CSV file? (yes/no): ").strip().lower() == "yes":
        file_name = input("Enter the filename (without extension): ").strip()
        save_data_to_csv(
            r"G:\My Drive\Determination_fluid_temp\Measurements\251007_MQ_ETH_Diff_temp",
            file_name,
            np.array(all_time),
            np.array(all_current),
            np.array(all_voltage),
            resistance,
            power,
            temperature,
            pulse_state,
            params
        )

    smu.close()


if __name__ == "__main__":
    main()
