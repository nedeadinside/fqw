from __future__ import annotations

import logging
from pathlib import Path

METRICS_ERROR_LOGGER = logging.getLogger("fqw.evaluation.metrics_errors")
METRICS_ERROR_LOGGER.addHandler(logging.NullHandler())
METRICS_ERROR_LOGGER.propagate = False


def _remove_file_handlers() -> None:
    for handler in list(METRICS_ERROR_LOGGER.handlers):
        if isinstance(handler, logging.FileHandler):
            METRICS_ERROR_LOGGER.removeHandler(handler)
            handler.close()


def configure_metrics_error_log(path: str | None) -> None:
    _remove_file_handlers()

    if not path:
        return

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    METRICS_ERROR_LOGGER.addHandler(file_handler)


def close_metrics_error_log() -> None:
    _remove_file_handlers()


def log_metric_error(message: str, *args) -> None:
    METRICS_ERROR_LOGGER.error(message, *args)
