"""Local paths, atomic output, dataset snapshots, and numeric summaries."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKDIR = PROJECT_ROOT / "work"
MODEL = "gpt-5.6-luna"
PREFIX = "llm1_gpt56luna"
# USD per million tokens; configurable through the CLI. Estimates, not billing.
# https://developers.openai.com/api/docs/models/gpt-5.6-luna (checked 2026-10-02)
INPUT_PRICE = 0.20
CACHED_INPUT_PRICE = 0.02
OUTPUT_PRICE = 1.20


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_clean(value):
    """Use JSON null, never nonstandard NaN/Infinity, in structured diagnostics."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    return value


def json_dumps(value):
    return json.dumps(json_clean(value), ensure_ascii=False, allow_nan=False)


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def population_variance(draws):
    valid = [float(d) for d in draws if d is not None and math.isfinite(float(d))]
    if not valid:
        return float("nan")
    mean = math.fsum(valid) / len(valid)
    return math.fsum((x - mean) ** 2 for x in valid) / len(valid)


def estimate_cost(usage, input_price=INPUT_PRICE, output_price=OUTPUT_PRICE,
                  cached_input_price=CACHED_INPUT_PRICE):
    cached = usage.get("cached_input_tokens", 0)
    return ((usage.get("input_tokens", 0) - cached) * input_price
            + cached * cached_input_price
            + usage.get("output_tokens", 0) * output_price) / 1_000_000


def load_benchmark(name, workdir=DEFAULT_WORKDIR):
    """Snapshot the full original preprocessing before applying any CLI limit.

    The JSON input snapshot freezes source ordering and text across resumes.
    """
    from .benchmarks import BENCHMARKS, LOADER_SOURCE_SHA256

    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"{PREFIX}_inputs_{name}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["loader_hash"] != LOADER_SOURCE_SHA256:
            raise ValueError("Dataset loader changed; use a new --output-dir.")
        frame = pd.DataFrame(saved["records"])
    else:
        frame = BENCHMARKS[name]["loader"]().reset_index(drop=True)
        atomic_write(path, json_dumps({"benchmark": name, "created_at": utc_now(),
                     "loader_hash": LOADER_SOURCE_SHA256,
                     "records": frame.to_dict(orient="records")}))
    if frame.item_id.isna().any() or frame.item_id.duplicated().any():
        raise ValueError(f"Missing or duplicate item IDs in {name}")
    if not frame.text.map(lambda x: isinstance(x, str)).all():
        raise ValueError(f"Non-text input in {name}")
    frame["item_id"] = frame.item_id.astype(str)
    return frame
