from typing import List, Dict, Any
from .utils import atomic_write_bytes, atomic_write_json
import io, csv
from math import isfinite

def save_csv(path: str, meta: Dict[str, Any], rows: List[Dict[str, Any]]):
    buf = io.StringIO()
    buf.write("# METADATA SETTINGS\n")
    for k, v in meta.items():
        buf.write(f"# {k}: {v}\n")
    buf.write("# ----------------\n")

    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["time_s", "current_a", "voltage_v", "resistance_ohm", "temperature_c", "pulse_active"])

    for r in rows:
        def fmt(x, fmtstr):
            try:
                if x is None: return ""
                f = float(x)
                if not isfinite(f): return ""
                return format(f, fmtstr)
            except Exception:
                return ""
        w.writerow([
            fmt(r.get("t_s"), ".6f"),
            fmt(r.get("i_a"), ".9f"),
            fmt(r.get("v_v"), ".9f"),
            fmt(r.get("r_ohm"), ".9f"),
            fmt(r.get("temp_c"), ".6f"),
            int(bool(r.get("pulse_active", False)))
        ])
    atomic_write_bytes(path, buf.getvalue().encode("utf-8"))

def save_json(path: str, payload: Dict[str, Any]):
    atomic_write_json(path, payload)
