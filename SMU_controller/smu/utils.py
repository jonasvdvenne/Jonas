import os, tempfile, json, time
from math import isfinite
import numpy as np

DEFAULT_ALPHA = 0.00381
DEFAULT_ROOM_TEMP_C = 22.0

def safe_float(val, default):
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default

def calculate_temperature(r, r0, alpha=DEFAULT_ALPHA, t0=DEFAULT_ROOM_TEMP_C):
    if r0 is None or r0 <= 0 or not isfinite(r0) or r is None or r <= 0 or not isfinite(r):
        return np.nan
    return (r / r0 - 1.0) / alpha + t0

def monotonic_s():
    # Use a monotonic clock for timing (robust against system clock jumps)
    return time.monotonic()

def atomic_write_bytes(path: str, data: bytes):
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=os.path.splitext(path)[1])
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass

def atomic_write_json(path: str, obj, indent=2):
    atomic_write_bytes(path, json.dumps(obj, indent=indent).encode("utf-8"))

class Rolling:
    """Simple rolling mean for temperature filtering."""
    def __init__(self, window: int):
        self.window = int(max(1, window))
        self.buf = []

    def push(self, x):
        self.buf.append(x)
        if len(self.buf) > self.window:
            self.buf = self.buf[-self.window:]

    def mean(self):
        vals = [v for v in self.buf if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan
