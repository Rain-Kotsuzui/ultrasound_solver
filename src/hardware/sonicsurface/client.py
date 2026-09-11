"""Client for submitting exported solver phases to the local hardware service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


def request_json(
    service_url: str,
    method: str,
    path: str,
    payload: dict | None = None,
) -> dict:
    data = (
        json.dumps(payload, allow_nan=False).encode("utf-8")
        if payload is not None
        else None
    )
    request = Request(
        f"{service_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def load_exported_phases(
    phase_file: str | Path,
    manifest_file: str | Path | None = None,
) -> np.ndarray:
    phase_file = Path(phase_file)
    phases = np.load(phase_file, allow_pickle=False)
    phases = np.asarray(phases, dtype=np.float64)
    if phases.ndim != 1 or not np.isfinite(phases).all():
        raise ValueError("Phase export must be a finite one-dimensional .npy vector")

    if manifest_file is None:
        stem = phase_file.name.removesuffix("_best_phases_rad.npy")
        manifest_file = phase_file.with_name(f"{stem}_phase_export.json")
    manifest_path = Path(manifest_file)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase_unit") != "rad":
        raise ValueError("Phase manifest must declare phase_unit='rad'")
    if manifest.get("hardware_ready") is not False:
        raise ValueError("Only continuous, non-device-ready solver exports are accepted")
    if int(manifest.get("num_transducers", -1)) != phases.size:
        raise ValueError("Manifest num_transducers does not match the phase vector")
    return phases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit exported phase vectors to the local SonicSurface service."
    )
    parser.add_argument("--service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--phase-file")
    parser.add_argument("--manifest-file")
    parser.add_argument("--off", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    commands = sum((bool(args.phase_file), args.off, args.status))
    if commands != 1:
        parser.error("Specify exactly one of --phase-file, --off, or --status")
    if args.status:
        print(json.dumps(request_json(args.service_url, "GET", "/v1/status"), indent=2))
        return
    if args.off:
        print(json.dumps(request_json(args.service_url, "POST", "/v1/off", {}), indent=2))
        return

    phases = load_exported_phases(args.phase_file, args.manifest_file)
    response = request_json(
        args.service_url,
        "POST",
        "/v1/pattern",
        {"phases_rad": phases.tolist()},
    )
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
