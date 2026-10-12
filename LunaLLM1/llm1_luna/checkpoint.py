"""Durable per-response checkpoints; CSV is an atomic, rebuildable export."""
from __future__ import annotations

import fcntl
import json
import sqlite3
import threading
from pathlib import Path

import pandas as pd

from .utils import atomic_write, json_dumps, utc_now

OUTPUT_COLUMNS = [
    "item_id", "benchmark", "llm1_label", "human_label", "human_uncertainty",
    "n_human_annotations", "resampling_var", "logprob_var",
    "valid_label_prob_mass", "n_valid_resamples", "status", "resampling_draws", "label_probabilities",
    "logprob_status", "input_tokens", "output_tokens", "cached_input_tokens",
    "total_requests", "successful_responses", "n_resamples_completed",
    "model", "created_at", "updated_at", "prompt_hash", "input_hash", "error",
]
HUMAN_COLUMNS = ("human_label", "human_uncertainty", "n_human_annotations")
OUTPUT_SCHEMA_VERSION = 2


def read_result(result):
    """Read old checkpoints at one boundary; new results use only the final names.

    Early Luna exports duplicated these three columns. Preserve their canonical
    values when reading an old checkpoint, then discard the redundant aliases.
    Raw API-response payloads are separate and retain their parsed `label` field.
    """
    result = dict(result)
    for alias, canonical in (("label", "llm1_label"), ("g_m", "resampling_var"),
                             ("n_valid_draws", "n_valid_resamples")):
        if alias in result:
            if canonical in result and result[canonical] != result[alias]:
                raise ValueError(f"Conflicting checkpoint columns: {alias} and {canonical}.")
            result.setdefault(canonical, result[alias])
            del result[alias]
    return result


class Checkpoint:
    def __init__(self, stem, config, metadata):
        self.stem = Path(stem)
        self.stem.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.file_lock = self.stem.with_suffix(".lock").open("a+")
        try:
            fcntl.flock(self.file_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file_lock.close()
            raise RuntimeError("This benchmark already has a running writer.") from None
        self.db = sqlite3.connect(self.stem.with_suffix(".sqlite3"), check_same_thread=False, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT PRIMARY KEY, input_hash TEXT NOT NULL, result TEXT);
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, item_id TEXT NOT NULL, slot INTEGER NOT NULL,
                started_at TEXT NOT NULL, status TEXT NOT NULL, payload TEXT, error TEXT);
            CREATE INDEX IF NOT EXISTS item_attempts ON attempts(item_id, slot);
        """)
        try:
            row = self.db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
            if row and json.loads(row[0]) != config:
                raise ValueError("Existing checkpoint configuration differs; use a new --output-dir.")
            if not row:
                meta_path = self.stem.with_suffix(".metadata.json")
                csv_path = self.stem.with_suffix(".csv")
                if csv_path.exists():
                    if not meta_path.exists() or json.loads(meta_path.read_text())["config"] != config:
                        raise ValueError("Existing CSV lacks matching provenance; use a new --output-dir.")
                    self._import_completed_csv(csv_path)
                with self.db:
                    self.db.execute("INSERT INTO metadata VALUES ('config', ?)", (json_dumps(config),))
                    self.db.execute("INSERT INTO metadata VALUES ('provenance', ?)", (json_dumps(metadata),))
            self.config = config
            self.metadata = json.loads(self.db.execute(
                "SELECT value FROM metadata WHERE key='provenance'").fetchone()[0])
            self._load_usage_counters()
        except BaseException:
            self.close()
            raise

    def _import_completed_csv(self, path):
        """Completed CSV rows also prevent repeat charges if the DB was moved away."""
        from .benchmarks import BENCHMARKS, R
        frame = pd.read_csv(path, dtype={"item_id": str}, keep_default_na=False)
        # Human columns are an additive export field; legacy Luna CSVs can still
        # recover their completed annotations and be enriched from source inputs.
        required = set(OUTPUT_COLUMNS) - set(HUMAN_COLUMNS)
        if not required.issubset(frame.columns):
            raise ValueError("Existing CSV schema is incomplete; preserve its SQLite checkpoint.")
        for result in frame.to_dict(orient="records"):
            if result["status"] != "complete":
                continue
            result = read_result(result)
            for key in HUMAN_COLUMNS:
                value = result.get(key, "")
                result[key] = None if value == "" else float(value) if key == "human_uncertainty" else int(float(value))
            for key in ("resampling_draws", "label_probabilities"):
                result[key] = json.loads(result[key])
            for key in ("llm1_label", "n_valid_resamples",
                        "input_tokens", "output_tokens", "cached_input_tokens", "total_requests",
                        "successful_responses", "n_resamples_completed"):
                result[key] = int(float(result[key]))
            for key in ("resampling_var", "logprob_var", "valid_label_prob_mass"):
                result[key] = float(result[key]) if result[key] != "" else None
            levels = BENCHMARKS[result["benchmark"]]["levels"]
            draws = result["resampling_draws"]
            if (result["llm1_label"] not in levels or result["n_resamples_completed"] != R
                    or len(draws) != R or any(x is not None and x not in levels for x in draws)
                    or sum(x is not None for x in draws) != result["n_valid_resamples"]):
                raise ValueError("Completed CSV row has invalid labels or resample count.")
            self.db.execute("INSERT INTO items VALUES (?, ?, ?)",
                            (result["item_id"], result["input_hash"], json_dumps(result)))
        self.db.commit()

    def ensure_item(self, item_id, input_hash):
        self.ensure_items([(item_id, input_hash)])

    def ensure_items(self, items):
        """Register inputs in one transaction before any billable requests."""
        items = list(items)
        with self.lock, self.db:
            existing = dict(self.db.execute("SELECT item_id,input_hash FROM items"))
            for item_id, input_hash in items:
                if item_id in existing and existing[item_id] != input_hash:
                    raise ValueError(f"Input changed for {item_id}; use a new --output-dir.")
            self.db.executemany("INSERT OR IGNORE INTO items VALUES (?, ?, NULL)", items)

    def get_result(self, item_id):
        with self.lock:
            row = self.db.execute("SELECT result FROM items WHERE item_id=?", (item_id,)).fetchone()
            return read_result(json.loads(row[0])) if row and row[0] else None

    def start_attempt(self, item_id, slot):
        with self.lock:
            with self.db:
                cursor = self.db.execute("INSERT INTO attempts(item_id,slot,started_at,status) VALUES (?,?,?,'pending')",
                                         (item_id, slot, utc_now()))
            self._add_usage(item_id, {"total_requests": 1})
            return cursor.lastrowid

    def record_error(self, attempt_id, error):
        with self.lock, self.db:
            self.db.execute("UPDATE attempts SET status='error',error=? WHERE id=?", (error, attempt_id))

    def record_response(self, attempt_id, result):
        with self.lock:
            row = self.db.execute("SELECT item_id,status FROM attempts WHERE id=?", (attempt_id,)).fetchone()
            if row is None or row[1] == "success":
                raise ValueError("Response must refer to one unfinished request attempt.")
            with self.db:
                self.db.execute("UPDATE attempts SET status='success',payload=? WHERE id=?",
                                (json_dumps(result), attempt_id))
            self._add_usage(row[0], {"successful_responses": 1,
                **{key: result.get(key, 0) or 0 for key in
                   ("input_tokens", "output_tokens", "cached_input_tokens")}})

    def calls(self, item_id):
        with self.lock:
            rows = self.db.execute("SELECT slot,payload FROM attempts WHERE item_id=? AND status='success' ORDER BY id",
                                   (item_id,)).fetchall()
            return {row[0]: json.loads(row[1]) for row in rows}

    @staticmethod
    def _empty_usage():
        return dict(input_tokens=0, output_tokens=0, cached_input_tokens=0,
                    total_requests=0, successful_responses=0)

    def _add_usage(self, item_id, delta):
        item = self._usage_by_item.setdefault(item_id, self._empty_usage())
        for key, value in delta.items():
            item[key] += value
            self._usage_total[key] += value

    def _load_usage_counters(self):
        """Rebuild once from the durable ledger, then update in constant time.

        Reading every historical response after every item is quadratic for a
        full corpus. These in-memory counters are never the durability source.
        """
        self._usage_by_item = {}
        self._usage_total = self._empty_usage()
        for row in self.db.execute("SELECT item_id,status,payload FROM attempts"):
            delta = {"total_requests": 1}
            if row[1] == "success":
                payload = json.loads(row[2])
                delta["successful_responses"] = 1
                delta.update({key: payload.get(key, 0) or 0 for key in
                              ("input_tokens", "output_tokens", "cached_input_tokens")})
            self._add_usage(row[0], delta)
        for row in self.db.execute("""SELECT item_id,result FROM items
                WHERE result IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM attempts WHERE attempts.item_id=items.item_id)"""):
            result = json.loads(row[1])
            self._add_usage(row[0], {key: result.get(key, 0) for key in self._usage_total})

    def usage(self, item_id=None):
        with self.lock:
            usage = self._usage_total if item_id is None else self._usage_by_item.get(item_id, self._empty_usage())
            return dict(usage)

    def save_result(self, result):
        self.save_results([result])

    def save_results(self, results):
        with self.lock, self.db:
            self.db.executemany("UPDATE items SET result=? WHERE item_id=?",
                               [(json_dumps(result), result["item_id"]) for result in results])

    def export(self):
        with self.lock:
            rows = self.db.execute("SELECT result FROM items WHERE result IS NOT NULL ORDER BY rowid").fetchall()
            results = [read_result(json.loads(row[0])) for row in rows]
            serial = []
            for result in results:
                result = dict(result)
                for column in ("resampling_draws", "label_probabilities"):
                    result[column] = json_dumps(result[column])
                serial.append(result)
            frame = pd.DataFrame(serial, columns=OUTPUT_COLUMNS)
            atomic_write(self.stem.with_suffix(".csv"), frame.to_csv(index=False))
            atomic_write(self.stem.with_suffix(".metadata.json"), json_dumps({
                **self.metadata, "config": self.config, "updated_at": utc_now(),
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "output_columns": OUTPUT_COLUMNS,
                "human_column_sources": {"human_label": "consensus", "human_uncertainty": "u_m",
                                         "n_human_annotations": "n_ann"},
                "usage": self.usage(), "items_exported": len(results),
                "items_complete": sum(r["status"] == "complete" for r in results),
                "item_errors": sum(r["status"] in ("api_error", "malformed_output") for r in results)}))
            return frame

    def close(self):
        self.db.close()
        fcntl.flock(self.file_lock.fileno(), fcntl.LOCK_UN)
        self.file_lock.close()
