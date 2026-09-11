"""Compile an exported solver phase vector into inspectable UART-frame JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import load_exported_phases
from .profile import load_profile
from .protocol import compile_pattern


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run SonicSurface frame export; never opens a serial port."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--phase-file", required=True)
    parser.add_argument("--manifest-file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    phases = load_exported_phases(args.phase_file, args.manifest_file)
    pattern = compile_pattern(phases, profile)
    output = {
        "schema_version": 1,
        "profile": profile.name,
        "protocol": profile.protocol,
        "device_bins": pattern.device_bins.tolist(),
        "frames_hex": [frame.hex() for frame in pattern.frames],
        "payload_sha256": pattern.sha256,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Dry-run frame export written to {path}")


if __name__ == "__main__":
    main()
