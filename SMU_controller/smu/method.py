# method.py
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import yaml
import threading, time

CONFIG_DIR = Path(__file__).resolve().parents[1] / "methods"

class MethodStep(BaseModel):
    target_temp_c: float
    duration_s: float
    pulses: bool = False
    pulse_power_w: Optional[float] = None
    pulse_duration_s: Optional[float] = None
    pulse_period_s: Optional[float] = None

class Method(BaseModel):
    name: str
    steps: List[MethodStep]

# ----------------- YAML IO -----------------

def list_methods() -> dict[str, Path]:
    CONFIG_DIR.mkdir(exist_ok=True, parents=True)
    out = {}
    for p in CONFIG_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(p.read_text())
            nm = data.get("name", p.stem)
            out[nm] = p
        except Exception:
            pass
    return out

def load_method(name_or_path: str | Path) -> Method:
    p = Path(name_or_path)
    if not p.exists():
        options = list_methods()
        if name_or_path in options:
            p = options[name_or_path]
        else:
            raise FileNotFoundError(f"Method {name_or_path!r} not found.")
    data = yaml.safe_load(p.read_text())
    return Method(**data)

def save_method(method: Method, filename: str | None = None) -> Path:
    CONFIG_DIR.mkdir(exist_ok=True, parents=True)
    fname = filename or f"{method.name}.yaml"
    path = CONFIG_DIR / fname
    with path.open("w") as f:
        yaml.safe_dump(method.model_dump(), f, sort_keys=False)
    return path

# ----------------- Runner -----------------

class MethodRunner:
    def __init__(self, controller, method: Method):
        self.controller = controller
        self.method = method
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        for step in self.method.steps:
            if not self._running:
                break

            # set temperature
            self.controller.set_target_temp(step.target_temp_c)

            # set pulse state
            if step.pulses:
                self.controller.set_pulse(
                    power_w=step.pulse_power_w or 0,
                    duration_s=step.pulse_duration_s or 0,
                    period_s=step.pulse_period_s or 0
                )
                self.controller.pulse_manual_override = False  # force active
            else:
                self.controller.pulse_manual_override = True   # force paused

            # wait for step duration
            t0 = time.time()
            while self._running and (time.time() - t0 < step.duration_s):
                time.sleep(0.5)

        self._running = False
