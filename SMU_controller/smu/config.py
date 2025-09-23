from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

class PIDBand(BaseModel):
    abs_error_gt: float = Field(..., description="Activate when |err| > this (descending precedence)")
    Kp: float
    Ki: float
    Kd: float

class PIDProfile(BaseModel):
    name: str
    description: Optional[str] = None
    temp_filter_window: int = 10
    deriv_clip: float = 50.0
    int_clip: float = 1000.0
    post_pulse_cooldown_s: float = 1.0
    post_pulse_drop_limit: float = 0.02
    alpha: float = 0.00381    # 🔹 add this line
    bands: List[PIDBand] = Field(default_factory=list)


    def pick_gains(self, err_abs: float):
        # bands are assumed sorted descending by abs_error_gt
        for b in sorted(self.bands, key=lambda x: x.abs_error_gt, reverse=True):
            if err_abs > b.abs_error_gt:
                return b.Kp, b.Ki, b.Kd
        # default to the smallest band (closest)
        if self.bands:
            b = sorted(self.bands, key=lambda x: x.abs_error_gt)[0]
            return b.Kp, b.Ki, b.Kd
        return 0.0, 0.0, 0.0

def list_profiles() -> Dict[str, Path]:
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

def load_profile(name_or_path: str | Path) -> PIDProfile:
    p = Path(name_or_path)
    if not p.exists():
        # look up by name
        options = list_profiles()
        if name_or_path in options:
            p = options[name_or_path]
        else:
            raise FileNotFoundError(f"PID profile {name_or_path!r} not found.")
    data = yaml.safe_load(p.read_text())
    return PIDProfile(**data)

def save_profile(profile: PIDProfile, filename: str | None = None) -> Path:
    CONFIG_DIR.mkdir(exist_ok=True, parents=True)
    fname = filename or f"{profile.name}.yaml"
    path = CONFIG_DIR / fname
    with path.open("w") as f:
        yaml.safe_dump(profile.model_dump(), f, sort_keys=False)
    return path
