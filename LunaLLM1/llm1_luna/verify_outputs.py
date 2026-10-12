"""Offline audit of saved samples, top-logprobs, human aggregates, and call counts."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmarks import BENCHMARKS, R, TEMP_RESAMPLE
from .checkpoint import HUMAN_COLUMNS, OUTPUT_COLUMNS, OUTPUT_SCHEMA_VERSION
from .llm1_client import extract_logprobs
from .utils import DEFAULT_WORKDIR, PREFIX


def verify_output(benchmark, output_dir=DEFAULT_WORKDIR):
    stem = Path(output_dir) / f"{PREFIX}_{benchmark}"
    frame = pd.read_csv(stem.with_suffix(".csv"), dtype={"item_id": str})
    metadata = json.loads(stem.with_suffix(".metadata.json").read_text())
    if metadata.get("output_schema_version") == OUTPUT_SCHEMA_VERSION:
        assert list(frame.columns) == OUTPUT_COLUMNS
    spec = BENCHMARKS[benchmark]
    source_path = Path(output_dir) / f"{PREFIX}_inputs_{benchmark}.json"
    source = {r["item_id"]: r for r in json.loads(source_path.read_text())["records"]} if source_path.exists() else None
    summaries = []
    with closing(sqlite3.connect(f"file:{stem.with_suffix('.sqlite3')}?mode=ro", uri=True)) as db:
        for row in frame.itertuples():
            if row.status != "complete":
                continue
            draws = json.loads(row.resampling_draws)
            valid = [x for x in draws if x is not None]
            assert row.llm1_label in spec["levels"]
            assert all(x in spec["levels"] for x in valid)
            assert len(draws) == R and len(valid) == row.n_valid_resamples
            assert math.isclose(row.resampling_var, float(np.var(valid)), abs_tol=1e-12)
            if source is not None and set(HUMAN_COLUMNS).issubset(frame.columns):
                human = source[row.item_id]
                for exported, original in (("human_label", "consensus"),
                                           ("human_uncertainty", "u_m"),
                                           ("n_human_annotations", "n_ann")):
                    if original in human:
                        assert math.isclose(float(getattr(row, exported)), float(human[original]), abs_tol=1e-12)
            calls = {slot: json.loads(payload) for slot, payload in db.execute(
                "SELECT slot,payload FROM attempts WHERE item_id=? AND status='success' ORDER BY id",
                (row.item_id,))}
            if not calls:
                # A CSV-only recovery has no raw responses to audit.
                summaries.append({"item_id": row.item_id, "call_audit": "CSV-only; raw responses unavailable"})
                continue
            assert set(calls) == set(range(R + 1))
            assert calls[0]["temperature"] == 0
            assert all(calls[i]["temperature"] == TEMP_RESAMPLE for i in range(1, R + 1))
            assert sum(call["logprobs_requested"] for call in calls.values()) == 1
            assert calls[1]["logprobs_requested"]
            assert row.llm1_label == calls[0]["label"]
            assert draws == [calls[i]["label"] for i in range(1, R + 1)]
            lp = extract_logprobs({"output": calls[1]["output_structure"]}, spec["levels"],
                                  metadata["config"]["min_valid_label_mass"])
            assert math.isclose(row.valid_label_prob_mass, lp["valid_label_prob_mass"], abs_tol=1e-12)
            probabilities = {int(k): v for k, v in json.loads(row.label_probabilities).items()}
            if lp["logprob_var"] is None:
                assert math.isnan(row.logprob_var) and not probabilities
            else:
                assert math.isclose(sum(probabilities.values()), 1, abs_tol=1e-12)
                mean = sum(k * p for k, p in probabilities.items())
                variance = sum(p * (k - mean) ** 2 for k, p in probabilities.items())
                assert math.isclose(row.logprob_var, variance, abs_tol=1e-12)
                assert math.isclose(row.logprob_var, lp["logprob_var"], abs_tol=1e-12)
                for k, probability in probabilities.items():
                    assert math.isclose(probability, lp["label_probabilities"][k], abs_tol=1e-12)
            summaries.append(dict(item_id=row.item_id, llm1_label=int(row.llm1_label),
                                  resampling_draws=draws, resampling_var=row.resampling_var,
                                  label_probabilities=probabilities,
                                  valid_label_prob_mass=row.valid_label_prob_mass,
                                  logprob_var=None if math.isnan(row.logprob_var) else row.logprob_var,
                                  logprob_status=row.logprob_status,
                                  n_valid_resamples=int(row.n_valid_resamples),
                                  successful_slots=len(calls), call_audit="passed"))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORKDIR)
    args = parser.parse_args()
    print(json.dumps(verify_output(args.benchmark, args.output_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
