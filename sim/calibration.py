from __future__ import annotations

import json
from pathlib import Path

from sim.kart_dynamics import KartDynamicsParams


DEFAULT_PARAMS_PATH = Path(__file__).with_name("kart_params.json")


def load_kart_params(params_path: str | Path | None = None) -> KartDynamicsParams:
    path = Path(params_path) if params_path is not None else DEFAULT_PARAMS_PATH
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return KartDynamicsParams.from_mapping(payload)


def save_kart_params(params_path: str | Path, payload: dict) -> Path:
    path = Path(params_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path

