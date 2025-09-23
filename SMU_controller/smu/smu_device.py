import time
from typing import Optional
import pyvisa

class KeysightB2901:
    """Thin wrapper over VISA resource with the exact SCPI used by your loop."""
    def __init__(self, resource: str | None = None):
        rm = pyvisa.ResourceManager()
        self.inst = None
        if resource:
            self.inst = rm.open_resource(resource)
        else:
            # try to auto-discover
            for addr in rm.list_resources():
                try:
                    i = rm.open_resource(addr)
                    idn = i.query("*IDN?").strip()
                    if "KEYSIGHT" in idn.upper() and "B2901" in idn.upper():
                        self.inst = i
                        break
                    i.close()
                except Exception:
                    pass
        if self.inst is None:
            raise RuntimeError("No Keysight B2901 found.")

        self.inst.timeout = 5000
        self.inst.write_termination = '\n'
        self.inst.read_termination = '\n'
        self.reset_and_configure()

    def reset_and_configure(self):
        w = self.inst.write
        w("*RST"); w("*CLS")
        w(":SOUR:FUNC:MODE CURR")
        w(":SENS:CURR:PROT 1")
        w("SOUR:CURR 0.0001")
        w("SOUR:CURR:RANG 0.3")
        w("SENS:VOLT:PROT 2")
        w("SENS:VOLT:RANG 2")
        w("SENS:REM ON")
        w(":SENS:VOLT:APER 0.001")
        w("SENS:CURR:RANG 0.4")
        w(":SENS:CURR:APER 0.001")
        w(":SENS:FUNC:CONC ON")
        w(":SENS:VOLT:NPLC 0.01")
        w(":SENS:CURR:NPLC 0.01")
        w(":SYST:AZER OFF")
        w(":SENS:AVER:STAT OFF")
        w(":DISP:ENAB ON")
        
    def set_ranges(self, current_range, voltage_range):
        # Map dropdown text → instrument commands
        curr_map = {
            "Auto": "AUTO",
            "10 µA": "1E-5",
            "100 µA": "1E-4",
            "1 mA": "1E-3",
            "10 mA": "1E-2",
            "100 mA": "0.1",
            "1 A": "1"
        }
        volt_map = {
            "Auto": "AUTO",
            "20 mV": "0.02",
            "200 mV": "0.2",
            "2 V": "2",
            "20 V": "20",
            "200 V": "200"
        }
    
        if current_range in curr_map:
            self.inst.write(f"SOUR:CURR:RANG {curr_map[current_range]}")
        if voltage_range in volt_map:
            self.inst.write(f"SENS:VOLT:RANG {volt_map[voltage_range]}")

    # Convenience passthroughs
    def write(self, scpi: str): self.inst.write(scpi)
    def query(self, scpi: str) -> str: return self.inst.query(scpi)
    def read(self) -> str: return self.inst.read()
    def close(self): 
        try: self.inst.write("OUTP OFF")
        except: pass
        try: self.inst.close()
        except: pass

class MockSMU:
    """
    Mock for development without hardware.
    Emulates a resistive load with temperature coefficient to produce V/I.
    """
    def __init__(self, r0=12.0, alpha=0.00381, t0=22.0):
        self._i_set = 1e-4
        self._r0 = r0
        self._alpha = alpha
        self._t = t0
        self._t0 = t0
        self._v = self._i_set * self._r0
        self._outp = False

    def write(self, scpi: str):
        cmd = scpi.strip().upper()
        if cmd.startswith("SOUR:CURR "):
            self._i_set = float(scpi.split()[-1])
            # simple thermal model: temp drifts towards 30–50 C based on I^2 R
            power = self._i_set**2 * self._r0
            self._t += 0.02 * (power * 20 - (self._t - self._t0))  # very crude
            r = self._r0 * (1 + self._alpha * (self._t - self._t0))
            self._v = self._i_set * r
        elif cmd == "OUTP ON":
            self._outp = True
        elif cmd == "OUTP OFF":
            self._outp = False
        # ignore the rest

    def query(self, scpi: str) -> str:
        cmd = scpi.strip().upper()
        if cmd.startswith("MEAS:CURR?;:MEAS:VOLT?"):
            return f"{self._i_set},{self._v}"
        elif cmd.startswith("*IDN?"):
            return "MOCK,SMU,0000,0.0"
        elif cmd.startswith("MEAS:CURR?"):
            return f"{self._i_set}"
        elif cmd.startswith("MEAS:VOLT?"):
            return f"{self._v}"
        return "0"
    
    def set_ranges(self, current_range, voltage_range):
        print(f"[MockSMU] Current range set to {current_range}, Voltage range set to {voltage_range}")

    def read(self) -> str:
        return "0"

    def close(self):
        self._outp = False
