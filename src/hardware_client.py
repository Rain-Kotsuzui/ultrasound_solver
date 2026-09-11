"""向本地 Ultraino 服务提交优化相位，不直接访问串口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


def request_json(url: str, method: str, path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise SystemExit(exc.read().decode("utf-8")) from exc
    except URLError as exc:
        raise SystemExit(f"无法连接本地硬件服务：{exc.reason}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="提交 Ultraino 256 通道优化相位。")
    parser.add_argument("--service-url", default="http://127.0.0.1:8765")
    parser.add_argument("--phase-file", type=Path)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--off", action="store_true")
    args = parser.parse_args()
    actions = int(args.phase_file is not None) + int(args.status) + int(args.off)
    if actions != 1:
        parser.error("必须且只能指定 --phase-file、--status 或 --off 之一")
    if args.status:
        print(json.dumps(request_json(args.service_url, "GET", "/v1/status"), indent=2))
    elif args.off:
        print(json.dumps(request_json(args.service_url, "POST", "/v1/off", {}), indent=2))
    else:
        phases = np.asarray(np.load(args.phase_file), dtype=np.float64)
        if phases.shape != (256,):
            parser.error("Ultraino SimpleFPGA 需要 256 个相位")
        print(
            json.dumps(
                request_json(
                    args.service_url,
                    "POST",
                    "/v1/pattern",
                    {"phases_rad": phases.tolist()},
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
