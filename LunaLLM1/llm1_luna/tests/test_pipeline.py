"""Offline behavioral tests. No credentials, network access, or API billing."""
import ast
import contextlib
import io
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from openai import APIConnectionError, AuthenticationError

from llm1_luna.benchmarks import BENCHMARKS, R, TEMP_RESAMPLE
from llm1_luna.llm1_client import CallFailure, LunaAnnotator, extract_logprobs, load_environment, parse_label
from llm1_luna.run_llm1 import run_benchmark
from llm1_luna.utils import PROJECT_ROOT


def response(text="1", top=None, status="completed"):
    return {"id": "test_response", "status": status, "model": "gpt-5.6-luna",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text,
                "logprobs": [{"token": text, "logprob": math.log(.6),
                              "top_logprobs": top or [
                                  {"token": "1", "logprob": math.log(.6)},
                                  {"token": "0", "logprob": math.log(.4)}]}]}]}],
            "usage": {"input_tokens": 10, "output_tokens": 1}}


class FakeResponses:
    def __init__(self, interrupt_at=None):
        self.requests = []
        self.interrupt_at = interrupt_at

    def create(self, **kwargs):
        if self.interrupt_at is not None and len(self.requests) == self.interrupt_at:
            raise KeyboardInterrupt
        self.requests.append(kwargs)
        return response("1" if len(self.requests) % 2 else "0")


class PipelineTests(unittest.TestCase):
    def test_registry_matches_notebook_active_subset(self):
        nb = json.loads((PROJECT_ROOT / "10_all_benchmarks_v2.ipynb").read_text())
        ns = {"_lewidi": lambda *a: None, "load_convabuse": lambda: None,
              "load_ambistory": lambda: None, "load_mhs": lambda *a: None}
        exec(compile(ast.parse("".join(nb["cells"][6]["source"])), "registry", "exec"), ns)
        self.assertEqual(list(BENCHMARKS), ns["RUN"])
        for name, spec in BENCHMARKS.items():
            for field in ("ann", "levels", "construct", "verb"):
                self.assertEqual(spec[field], ns["BENCHMARKS"][name][field])
        self.assertEqual((R, TEMP_RESAMPLE), (20, 1.0))
        original = ast.parse("".join(nb["cells"][5]["source"]))
        extracted = ast.parse((PROJECT_ROOT / "llm1_luna/benchmarks.py").read_text())
        for node in ast.walk(extracted):
            if isinstance(node, ast.Call):
                node.keywords = [kw for kw in node.keywords if kw.arg != "timeout"]
        for name in ("_lewidi", "load_convabuse", "load_ambistory"):
            before = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
            after = next(n for n in extracted.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(before), ast.dump(after))

    def test_missing_key_fails_without_display(self):
        with patch.dict("os.environ", {}, clear=True), patch("llm1_luna.llm1_client.load_dotenv"):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY not found"):
                load_environment()

    def test_strict_labels(self):
        for text in ("1", " 1", "\n1\n"):
            self.assertEqual(parse_label(text, [0, 1]), 1)
        for text in ("10", "I chose 1", "1.0", "1 or 0", "", "2"):
            self.assertIsNone(parse_label(text, [0, 1]))

    def test_exponentiation_variants_and_normalization(self):
        top = [{"token": "1", "logprob": math.log(.2)},
               {"token": " 1", "logprob": math.log(.1)},
               {"token": "5", "logprob": math.log(.6)},
               {"token": "no", "logprob": math.log(.1)}]
        out = extract_logprobs(response(top=top), [1, 2, 3, 4, 5], .8)
        self.assertAlmostEqual(out["valid_label_prob_mass"], .9)
        self.assertAlmostEqual(sum(out["label_probabilities"].values()), 1)
        self.assertAlmostEqual(out["label_probabilities"][1], 1 / 3)
        self.assertAlmostEqual(out["logprob_var"], 32 / 9)
        self.assertEqual(out["logprob_diagnostics"]["missing_labels"], [2, 3, 4])

    def test_low_mass_and_missing_are_not_fabricated(self):
        low = extract_logprobs(response(top=[{"token": "1", "logprob": math.log(.1)}]), [0, 1])
        self.assertIsNone(low["logprob_var"])
        self.assertAlmostEqual(low["valid_label_prob_mass"], .1)
        self.assertEqual(low["label_probabilities"], {})
        none = extract_logprobs(response(top=[{"token": "x", "logprob": 0}]), [0, 1])
        self.assertEqual(none["logprob_diagnostics"]["status"], "no_valid_label_tokens")
        self.assertIsNone(extract_logprobs({"output": []}, [0, 1])["logprob_var"])

    def test_api_numeric_roundoff_preserves_raw_mass(self):
        out = extract_logprobs(response(top=[{"token": "1", "logprob": .000004}]), [0, 1])
        self.assertAlmostEqual(out["valid_label_prob_mass"], math.exp(.000004))
        self.assertEqual(out["logprob_var"], 0)
        self.assertEqual(out["label_probabilities"], {0: 0, 1: 1})
        self.assertTrue(out["logprob_diagnostics"]["numeric_roundoff_detected"])
        out = extract_logprobs(response(top=[{"token": "0", "logprob": -.020897},
                                             {"token": "1", "logprob": -3.878441}]), [0, 1])
        self.assertGreater(out["valid_label_prob_mass"], 1)
        self.assertAlmostEqual(sum(out["label_probabilities"].values()), 1)
        self.assertGreater(out["logprob_var"], 0)
        self.assertEqual(out["logprob_diagnostics"]["status"], "ok_numeric_roundoff")
        bad = extract_logprobs(response(top=[{"token": "1", "logprob": .001}]), [0, 1])
        self.assertIsNone(bad["logprob_var"])
        bad = extract_logprobs(response(top=[{"token": "0", "logprob": math.log(.6)},
                                             {"token": "1", "logprob": math.log(.6)}]), [0, 1])
        self.assertEqual(bad["logprob_diagnostics"]["status"], "invalid_probability_mass")

    def test_failed_commit_does_not_increment_usage(self):
        from llm1_luna.checkpoint import Checkpoint
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "ledger"
            store = Checkpoint(stem, {}, {})
            reader = sqlite3.connect(stem.with_suffix(".sqlite3"))
            try:
                self.assertEqual(store.db.execute("PRAGMA busy_timeout").fetchone()[0], 60000)
                store.ensure_item("item", "hash")
                store.db.execute("PRAGMA busy_timeout=1")
                reader.execute("BEGIN")
                reader.execute("SELECT * FROM items").fetchall()
                with self.assertRaises(sqlite3.OperationalError):
                    store.start_attempt("item", 0)
                self.assertEqual(store.usage()["total_requests"], 0)
                reader.rollback()
                store.start_attempt("item", 0)
                self.assertEqual(store.usage()["total_requests"], 1)
            finally:
                reader.close()
                store.close()

    def test_progress_monitor_does_not_open_sqlite(self):
        from llm1_luna.paper_run import read_progress
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "llm1_gpt56luna_HSBrexit.metadata.json").write_text(json.dumps({
                "items_complete": 25, "item_errors": 0, "updated_at": "test",
                "usage": {"total_requests": 525, "successful_responses": 525}}))
            with patch("sqlite3.connect", side_effect=AssertionError("Monitor must not open SQLite")):
                progress = read_progress(p)
            self.assertEqual(progress["HSBrexit"]["completed_items"], 25)
            self.assertEqual(progress["HSBrexit"]["recorded_requests"], 525)

    def test_no_shift_to_later_digit(self):
        payload = response()
        logs = payload["output"][0]["content"][0]["logprobs"]
        logs.insert(0, {"token": "\n", "top_logprobs": [{"token": "\n", "logprob": 0}]})
        self.assertIsNone(extract_logprobs(payload, [0, 1])["logprob_var"])

    def test_sample_count_partial_resume_and_csv_recovery(self):
        frame = pd.DataFrame([{"item_id": "test-1", "text": "fixture", "split_orig": "train"}])
        fake = FakeResponses(interrupt_at=5)
        client = LunaAnnotator(client=SimpleNamespace(responses=fake))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                run_benchmark("HSBrexit", frame=frame, annotator=client, output_dir=directory)
            self.assertEqual(len(fake.requests), 5)
            fake.interrupt_at = None
            result = run_benchmark("HSBrexit", frame=frame, annotator=client, output_dir=directory)
            self.assertEqual(len(fake.requests), 21)
            self.assertEqual([r["temperature"] for r in fake.requests], [0.0] + [1.0] * 20)
            self.assertEqual(sum(r.get("top_logprobs") == 20 for r in fake.requests), 1)
            self.assertEqual(fake.requests[1]["include"], ["message.output_text.logprobs"])
            self.assertTrue(all(r["reasoning"] == {"effort": "none"} for r in fake.requests))
            saved = result.iloc[0]
            draws = json.loads(saved.resampling_draws)
            self.assertEqual(len(draws), 20)
            self.assertAlmostEqual(saved.resampling_var, np.var(draws))
            self.assertEqual(saved.status, "complete")
            again = run_benchmark("HSBrexit", frame=frame, annotator=client, output_dir=directory)
            self.assertEqual(len(fake.requests), 21)
            self.assertEqual(again.attrs["run_report"]["this_run"]["total_requests"], 0)
            # Recovery from CSV alone must not charge completed items again.
            Path(directory, "llm1_gpt56luna_HSBrexit.sqlite3").rename(Path(directory, "saved.sqlite3"))
            run_benchmark("HSBrexit", frame=frame, annotator=client, output_dir=directory)
            self.assertEqual(len(fake.requests), 21)
            changed = frame.assign(text="different input")
            with self.assertRaises(ValueError):
                run_benchmark("HSBrexit", frame=changed, annotator=client, output_dir=directory)

    def test_invalid_draw_not_replaced_and_truncation_invalid(self):
        fake = FakeResponses()
        client = LunaAnnotator(client=SimpleNamespace(responses=fake))
        with patch.object(fake, "create", return_value=response("1", status="incomplete")):
            self.assertIsNone(client.annotate_once("x", 0, "HSBrexit")["label"])
        frame = pd.DataFrame([{"item_id": "bad-draw", "text": "fixture"}])
        original = fake.create
        def one_malformed(**params):
            result = original(**params)
            if len(fake.requests) == 3:
                result["output"][0]["content"][0]["text"] = "Label: 1"
            return result
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch.object(fake, "create", side_effect=one_malformed):
            result = run_benchmark("HSBrexit", frame=frame, annotator=client, output_dir=directory)
            draws = json.loads(result.iloc[0].resampling_draws)
            self.assertEqual(len(fake.requests), 21)
            self.assertEqual(result.iloc[0].n_valid_resamples, 19)
            self.assertIn(None, draws)
            self.assertAlmostEqual(result.iloc[0].resampling_var, np.var([x for x in draws if x is not None]))

    def test_retry_bounded_and_auth_fails_fast(self):
        fake = FakeResponses()
        client = LunaAnnotator(client=SimpleNamespace(responses=fake), max_attempts=2)
        attempts, errors = [], []
        callbacks = dict(before_attempt=lambda: attempts.append(1), after_error=errors.append)
        network_error = APIConnectionError(request=SimpleNamespace())
        with patch.object(fake, "create", side_effect=[network_error, response()]), patch("llm1_luna.llm1_client.time.sleep"):
            result = client.annotate_once("x", 0, "HSBrexit", **callbacks)
            self.assertEqual(result["label"], 1)
            self.assertEqual((len(attempts), len(errors)), (2, 1))
        attempts.clear()
        with patch.object(fake, "create", side_effect=network_error), patch("llm1_luna.llm1_client.time.sleep"):
            with self.assertRaises(CallFailure) as failure:
                client.annotate_once("x", 0, "HSBrexit", **callbacks)
            self.assertFalse(failure.exception.fatal)
            self.assertEqual(len(attempts), 2)
        attempts.clear()
        http_response = SimpleNamespace(status_code=401, request=SimpleNamespace(), headers={})
        auth_error = AuthenticationError("Do not persist an arbitrary server body", response=http_response, body=None)
        with patch.object(fake, "create", side_effect=auth_error):
            with self.assertRaises(CallFailure) as failure:
                client.annotate_once("x", 0, "HSBrexit", **callbacks)
            self.assertTrue(failure.exception.fatal)
            self.assertEqual(len(attempts), 1)
            self.assertNotIn("arbitrary server body", str(failure.exception))

    def test_transient_item_error_resumes_missing_slots(self):
        fake = FakeResponses()
        client = LunaAnnotator(client=SimpleNamespace(responses=fake), max_attempts=1)
        frame = pd.DataFrame([{"item_id": "retry", "text": "fixture"}])
        original = fake.create
        def failing(**params):
            if len(fake.requests) == 4:
                raise APIConnectionError(request=SimpleNamespace())
            return original(**params)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(fake, "create", side_effect=failing):
                result = run_benchmark("HSBrexit", frame=frame, annotator=client, output_dir=directory)
                self.assertEqual(result.iloc[0].status, "api_error")
            result = run_benchmark("HSBrexit", frame=frame, annotator=client, output_dir=directory)
            self.assertEqual(result.iloc[0].status, "complete")
            self.assertEqual(len(fake.requests), 21)
            self.assertEqual(result.iloc[0].successful_responses, 21)
            self.assertEqual(result.iloc[0].total_requests, 22)

    def test_paper_supervisor_uses_fresh_separate_outputs(self):
        from llm1_luna import paper_run
        from llm1_luna.benchmarks import LOADER_SOURCE_SHA256
        fake = FakeResponses()
        client = LunaAnnotator(client=SimpleNamespace(responses=fake))
        frame = pd.DataFrame([{"item_id": "paper-1", "text": "fixture"},
                              {"item_id": "paper-2", "text": "another fixture"}])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            smoke = root / "smoke"
            smoke.mkdir()
            (smoke / "llm1_gpt56luna_inputs_HSBrexit.json").write_text(json.dumps({
                "benchmark": "HSBrexit", "loader_hash": LOADER_SOURCE_SHA256,
                "records": frame.to_dict(orient="records")}))
            (smoke / "llm1_gpt56luna_HSBrexit.csv").write_text("smoke results must remain isolated")
            def run_with_fake(name, **kwargs):
                return run_benchmark(name, annotator=client, **kwargs)
            with patch.object(paper_run, "BENCHMARKS", {"HSBrexit": BENCHMARKS["HSBrexit"]}), \
                 patch.object(paper_run, "DEFAULT_WORKDIR", smoke), \
                 patch.object(paper_run, "load_environment", return_value=True), \
                 patch.object(paper_run.shutil, "which", return_value=None), \
                 patch.object(paper_run, "run_benchmark", side_effect=run_with_fake):
                paper = root / "paper"
                self.assertEqual(paper_run.run_paper(paper, workers=2), 0)
                self.assertEqual(len(fake.requests), 42)
                status = json.loads((paper / "paper_run_status.json").read_text())
                self.assertEqual(status["status"], "complete")
                self.assertFalse(status["smoke_annotations_reused"])
                self.assertEqual(status["benchmark_reports"]["HSBrexit"]["audited_items"], 2)
                self.assertEqual(status["total_usage"]["successful_responses"], 42)
                self.assertEqual(paper_run.run_paper(paper, workers=2), 0)
                self.assertEqual(len(fake.requests), 42)
                self.assertEqual((smoke / "llm1_gpt56luna_HSBrexit.csv").read_text(),
                                 "smoke results must remain isolated")

    def test_human_annotations_are_source_fields_not_llm_labels(self):
        fake = FakeResponses()
        client = LunaAnnotator(client=SimpleNamespace(responses=fake))
        frame = pd.DataFrame([{"item_id": "human-fields", "text": "fixture",
                               "consensus": 0, "u_m": .25, "n_ann": 4}])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = run_benchmark("HSBrexit", frame=frame, annotator=client, output_dir=directory)
            row = result.iloc[0]
            self.assertFalse({"label", "g_m", "n_valid_draws"} & set(result.columns))
            self.assertEqual(row.llm1_label, 1)
            self.assertEqual(row.human_label, 0)
            self.assertEqual(row.human_uncertainty, .25)
            self.assertEqual(row.n_human_annotations, 4)
            stem = Path(directory) / "llm1_gpt56luna_HSBrexit"
            # Legacy CSV recovery must add the human fields without re-annotating.
            old = pd.read_csv(stem.with_suffix(".csv"))
            old["label"] = old.llm1_label
            old["g_m"] = old.resampling_var
            old["n_valid_draws"] = old.n_valid_resamples
            old.drop(columns=["human_label", "human_uncertainty", "n_human_annotations"]).to_csv(
                stem.with_suffix(".csv"), index=False)
            stem.with_suffix(".sqlite3").rename(Path(directory) / "saved.sqlite3")
            again = run_benchmark("HSBrexit", frame=frame, annotator=client, output_dir=directory)
            self.assertEqual(len(fake.requests), 21)
            self.assertEqual(again.iloc[0].human_uncertainty, .25)
            self.assertEqual(again.iloc[0].human_label, 0)

    def test_offline_export_preserves_calls_and_joins_human_fields_by_id(self):
        from llm1_luna.export_results import export_completed
        fake = FakeResponses()
        client = LunaAnnotator(client=SimpleNamespace(responses=fake))
        frame = pd.DataFrame([
            {"item_id": "b", "text": "second", "consensus": 0, "u_m": .25, "n_ann": 4},
            {"item_id": "a", "text": "first", "consensus": 1, "u_m": 0., "n_ann": 3}])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            before = run_benchmark("HSBrexit", frame=frame, annotator=client, output_dir=directory)
            stem = Path(directory) / "llm1_gpt56luna_HSBrexit"
            snapshot = Path(directory) / "llm1_gpt56luna_inputs_HSBrexit.json"
            snapshot.write_text(json.dumps({"benchmark": "HSBrexit",
                                          "records": frame.iloc[::-1].to_dict(orient="records")}))
            # Represent an old result summary; raw response payloads stay untouched.
            with contextlib.closing(sqlite3.connect(stem.with_suffix(".sqlite3"))) as db, db:
                raw_calls = list(db.execute("SELECT * FROM attempts"))
                for item_id, payload in db.execute("SELECT item_id,result FROM items").fetchall():
                    result = json.loads(payload)
                    result.update(label=result["llm1_label"], g_m=result["resampling_var"],
                                  n_valid_draws=result["n_valid_resamples"])
                    for name in ("human_label", "human_uncertainty", "n_human_annotations"):
                        result.pop(name)
                    db.execute("UPDATE items SET result=? WHERE item_id=?", (json.dumps(result), item_id))
            with patch("llm1_luna.run_llm1.LunaAnnotator", side_effect=AssertionError("No API client")):
                report = export_completed("HSBrexit", directory)
            after = pd.read_csv(stem.with_suffix(".csv"))
            self.assertEqual(report["additional_api_requests"], 0)
            self.assertEqual(report["audited_items"], 2)
            self.assertEqual(report["usage_before"], report["usage_after"])
            self.assertEqual(after.set_index("item_id").human_label.to_dict(), {"b": 0, "a": 1})
            self.assertFalse({"label", "g_m", "n_valid_draws"} & set(after.columns))
            for name in ("llm1_label", "resampling_var", "logprob_var", "resampling_draws"):
                self.assertEqual(before[name].tolist(), after[name].tolist())
            with contextlib.closing(sqlite3.connect(stem.with_suffix(".sqlite3"))) as db:
                self.assertEqual(raw_calls, list(db.execute("SELECT * FROM attempts")))
            self.assertEqual(len(fake.requests), 42)


if __name__ == "__main__":
    unittest.main()
