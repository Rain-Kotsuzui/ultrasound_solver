"""服务端结构化审计日志。"""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JsonAuditFormatter(logging.Formatter):
    """将服务事件写成一行 JSON，避免记录完整相位向量。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "event": getattr(record, "event", "message"),
            "message": record.getMessage(),
        }
        details = getattr(record, "details", None)
        if details:
            payload.update(details)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_service_logging(
    log_file: str | Path,
    level: str = "INFO",
) -> logging.Logger:
    """配置轮转文件日志和控制台日志，并返回唯一服务 logger。"""
    logger = logging.getLogger("hardware.sonicsurface.service")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    logger.propagate = False

    formatter = JsonAuditFormatter()
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
