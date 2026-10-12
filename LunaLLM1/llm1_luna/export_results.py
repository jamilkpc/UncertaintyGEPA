"""Re-export completed benchmarks with human aggregates, without API requests.

The regular runner already writes this schema. This command upgrades existing
completed outputs using only their response checkpoint and frozen input snapshot.
It refuses active or incomplete benchmarks and records a compact audit report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .benchmarks import BENCHMARKS
from .checkpoint import Checkpoint, HUMAN_COLUMNS, OUTPUT_COLUMNS, OUTPUT_SCHEMA_VERSION
from .run_llm1 import human_fields
from .utils import DEFAULT_WORKDIR, PREFIX, atomic_write, digest, json_dumps, utc_now
from .verify_outputs import verify_output


def export_completed(benchmark, output_dir=DEFAULT_WORKDIR):
    """Preserve every model result; join source aggregates by exact item ID."""
    directory = Path(output_dir)
    stem = directory / f"{PREFIX}_{benchmark}"
    source_path = directory / f"{PREFIX}_inputs_{benchmark}.json"
    source_bytes = source_path.read_bytes()
    snapshot = json.loads(source_bytes)
    if snapshot["benchmark"] != benchmark:
        raise ValueError("Input snapshot benchmark does not match the requested export.")
    source = {str(row["item_id"]): row for row in snapshot["records"]}
    if not source or len(source) != len(snapshot["records"]):
        raise ValueError("Input snapshot is empty or contains duplicate item IDs.")
    metadata = json.loads(stem.with_suffix(".metadata.json").read_text())
    if metadata["items_complete"] != len(source):
        raise ValueError("Only a fully completed benchmark can be re-exported offline.")
    csv_path = stem.with_suffix(".csv")
    previous_csv_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    store = Checkpoint(stem, metadata["config"], metadata)
    try:
        before = store.usage()
        saved_ids = {row[0] for row in store.db.execute("SELECT item_id FROM items")}
        if saved_ids != set(source):
            raise ValueError("Checkpoint and source snapshot have different item IDs.")
        results = []
        for item_id, row in source.items():
            saved = store.get_result(item_id)
            if not saved or saved["status"] != "complete":
                raise ValueError(f"Incomplete checkpoint item: {item_id}")
            if saved["input_hash"] != digest(row["text"]):
                raise ValueError(f"Source text changed for item: {item_id}")
            if not {"consensus", "u_m", "n_ann"}.issubset(row):
                raise ValueError(f"Missing human aggregates for item: {item_id}")
            results.append({**saved, **human_fields(SimpleNamespace(**row))})
        store.save_results(results)
        frame = store.export()
        after = store.usage()
        assert before == after, "Offline export changed the API usage ledger."
        assert not frame[list(HUMAN_COLUMNS)].isna().any().any()
        audits = verify_output(benchmark, directory)
        assert len(audits) == len(source)
        report = dict(benchmark=benchmark, exported_at=utc_now(),
                      output_schema_version=OUTPUT_SCHEMA_VERSION,
                      rows=len(frame), audited_items=len(audits),
                      output_columns=OUTPUT_COLUMNS,
                      previous_csv_sha256=previous_csv_hash,
                      csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                      source_snapshot_sha256=hashlib.sha256(source_bytes).hexdigest(),
                      export_source_sha256={name: hashlib.sha256(
                          (Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("export_results.py", "checkpoint.py", "run_llm1.py", "verify_outputs.py")},
                      usage_before=before, usage_after=after, additional_api_requests=0)
        atomic_write(stem.with_suffix(".export_audit.json"), json_dumps(report))
        return report
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORKDIR)
    args = parser.parse_args()
    for benchmark in dict.fromkeys(args.benchmark):
        print(json_dumps(export_completed(benchmark, args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
