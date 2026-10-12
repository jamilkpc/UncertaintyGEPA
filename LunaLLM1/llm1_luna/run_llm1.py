"""Run only LLM1: python -m llm1_luna.run_llm1 --benchmark HSBrexit --limit 3."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import signal
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from . import __version__
from .benchmarks import (BENCHMARKS, DEFAULT_BENCHMARKS, LOADER_SOURCE_SHA256,
                         R, SEED, SOURCE_NOTEBOOK_SHA256, TEMP_RESAMPLE)
from .checkpoint import Checkpoint
from .llm1_client import CallFailure, LunaAnnotator, extract_logprobs, load_environment
from .utils import (CACHED_INPUT_PRICE, DEFAULT_WORKDIR, INPUT_PRICE, MODEL,
                    OUTPUT_PRICE, PREFIX, digest, estimate_cost, load_benchmark,
                    population_variance, utc_now)


def human_fields(row):
    """Copy the original human aggregates; never discretize or infer them."""
    label = getattr(row, "consensus", None)
    uncertainty = getattr(row, "u_m", None)
    n_annotations = getattr(row, "n_ann", None)
    return dict(human_label=int(label) if label is not None else None,
                human_uncertainty=float(uncertainty) if uncertainty is not None else None,
                n_human_annotations=int(n_annotations) if n_annotations is not None else None)


def _item_result(row, benchmark, store, error=""):
    calls = store.calls(row.item_id)
    canonical = calls.get(0, {})
    samples = [calls[i].get("label") for i in range(1, R + 1) if i in calls]
    lp = calls.get(1, {})
    if "output_structure" in lp:
        lp = extract_logprobs({"output": lp["output_structure"]}, BENCHMARKS[benchmark]["levels"],
                             store.config["min_valid_label_mass"])
    label = canonical.get("label")
    valid = sum(x is not None for x in samples)
    complete = label is not None and len(samples) == R and valid > 0
    status = "complete" if complete else ("api_error" if error else "incomplete")
    if not error and len(samples) == R and (label is None or not valid):
        status = "malformed_output"
    prior = store.get_result(row.item_id)
    variance = population_variance(samples)
    return dict(item_id=row.item_id, benchmark=benchmark, llm1_label=label,
                resampling_var=variance, logprob_var=lp.get("logprob_var"),
                valid_label_prob_mass=lp.get("valid_label_prob_mass"), n_valid_resamples=valid,
                status=status,
                resampling_draws=samples, label_probabilities=lp.get("label_probabilities", {}),
                logprob_status=lp.get("logprob_diagnostics", {}).get("status", "not_requested_yet"),
                **store.usage(row.item_id), n_resamples_completed=len(samples), model=MODEL,
                created_at=prior["created_at"] if prior else canonical.get("created_at", utc_now()),
                updated_at=utc_now(), prompt_hash=store.config["prompt_hash"],
                input_hash=digest(row.text), error=error, **human_fields(row))


def _process_item(row, benchmark, store, annotator, stop):
    cached = store.calls(row.item_id)
    fatal = None
    error = ""
    try:
        for slot in range(R + 1):
            # Invalid stochastic draws remain part of the original 20. A malformed
            # canonical response may be retried once on the next invocation.
            if slot in cached and (slot != 0 or cached[slot]["label"] is not None):
                continue
            if stop.is_set():
                break
            attempt_id = None

            def before_attempt():
                nonlocal attempt_id
                attempt_id = store.start_attempt(row.item_id, slot)

            def after_error(message):
                store.record_error(attempt_id, message)

            result = annotator.annotate_once(
                row.text, 0.0 if slot == 0 else TEMP_RESAMPLE, benchmark,
                logprobs=(slot == 1), before_attempt=before_attempt,
                after_error=after_error, stop_event=stop)
            store.record_response(attempt_id, result)
    except CallFailure as exc:
        error = str(exc)
        if exc.fatal:
            stop.set()
            fatal = exc
    finally:
        # The call ledger already protects every received response even if this
        # summary/export has not yet happened when the process is terminated.
        result = _item_result(row, benchmark, store, error)
        store.save_result(result)
    return result, fatal


def run_benchmark(benchmark, *, limit=None, workers=1, output_dir=DEFAULT_WORKDIR,
                  checkpoint_every=5, min_valid_mass=0.90, max_attempts=5, timeout=60.0,
                  input_price=INPUT_PRICE, output_price=OUTPUT_PRICE,
                  cached_input_price=CACHED_INPUT_PRICE, frame=None, annotator=None):
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if workers < 1 or checkpoint_every < 1 or (limit is not None and limit < 1):
        raise ValueError("workers, checkpoint-every, and limit must be positive.")
    if annotator is None:
        # Missing credentials fail before dataset downloads or output mutation.
        annotator = LunaAnnotator(timeout=timeout, max_attempts=max_attempts,
                                  min_valid_mass=min_valid_mass)
    spec = BENCHMARKS[benchmark]
    config = dict(schema_version=1, package_version=__version__, model=MODEL,
                  benchmark=benchmark, levels=spec["levels"], resamples=R,
                  canonical_temperature=0.0, resampling_temperature=TEMP_RESAMPLE,
                  reasoning_effort="none", top_logprobs=20, logprob_resample_index=0,
                  min_valid_label_mass=annotator.min_valid_mass,
                  max_output_tokens=annotator.max_output_tokens,
                  variance_ddof=0, prompt_hash=digest(spec["ann"]),
                  loader_hash=LOADER_SOURCE_SHA256, seed_for_dataset=SEED)
    metadata = dict(created_at=utc_now(), source_notebook="10_all_benchmarks_v2.ipynb",
                    source_notebook_sha256=SOURCE_NOTEBOOK_SHA256,
                    annotation_prompt=spec["ann"], user_template="ITEM:\n{text}",
                    openai_sdk_version=importlib.metadata.version("openai"),
                    numpy_version=importlib.metadata.version("numpy"),
                    pandas_version=importlib.metadata.version("pandas"),
                    prices_per_million=dict(input=input_price, output=output_price,
                                            cached_input=cached_input_price),
                    reproducibility="API outputs can vary even at temperature zero.")
    stem = Path(output_dir) / f"{PREFIX}_{benchmark}"
    store = Checkpoint(stem, config, metadata)
    stop = threading.Event()
    pool = None
    interrupted = False
    old_sigterm = None
    baseline = store.usage()
    selected_ids = set()
    try:
        frame = load_benchmark(benchmark, output_dir) if frame is None else frame.copy()
        if frame.item_id.isna().any() or frame.item_id.duplicated().any():
            raise ValueError("Missing or duplicate item IDs.")
        frame["item_id"] = frame.item_id.astype(str)
        selected = frame.head(limit) if limit is not None else frame
        selected_ids = set(selected.item_id)
        todo = []
        refreshed = []
        store.ensure_items((row.item_id, digest(row.text)) for row in selected.itertuples())
        for row in selected.itertuples():
            saved = store.get_result(row.item_id)
            if saved and saved["status"] == "complete":
                if store.calls(row.item_id):
                    # Refresh derived diagnostics and human fields without API calls.
                    refreshed.append(_item_result(row, benchmark, store))
                else:
                    # CSV-only recovery still has the original source input snapshot.
                    refreshed.append({**saved, **human_fields(row)})
            if not saved or saved["status"] != "complete":
                todo.append(row)
        store.save_results(refreshed)
        baseline = store.usage()
        print(f"{benchmark}: {len(selected)} selected, {len(selected)-len(todo)} complete, "
              f"{len(todo)} pending; up to {workers} concurrent requests.", flush=True)
        if threading.current_thread() is threading.main_thread():
            old_sigterm = signal.getsignal(signal.SIGTERM)
            def handle_sigterm(signum, sigframe):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, handle_sigterm)
        pool = ThreadPoolExecutor(max_workers=workers)
        iterator = iter(todo)
        pending = set()

        def submit_next():
            row = next(iterator, None)
            if row is not None and not stop.is_set():
                pending.add(pool.submit(_process_item, row, benchmark, store, annotator, stop))

        for _ in range(workers):
            submit_next()
        finished = 0
        fatal = None
        while pending:
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                result, failure = future.result()
                finished += 1
                fatal = fatal or failure
                usage = store.usage()
                print(f"  {benchmark} {finished}/{len(todo)} {result['item_id']}: "
                      f"{result['status']}; llm1_label={result['llm1_label']}, valid={result['n_valid_resamples']}/{R}; "
                      f"cumulative requests={usage['total_requests']}, "
                      f"tokens={usage['input_tokens']} in/{usage['output_tokens']} out", flush=True)
                if finished % checkpoint_every == 0:
                    store.export()
            for _ in ready:
                submit_next()
        if fatal:
            raise fatal
    except KeyboardInterrupt:
        interrupted = True
        stop.set()
        print("Interrupted: finishing in-flight requests and saving received responses.", flush=True)
    finally:
        stop.set()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        try:
            exported = store.export()
            total_usage = store.usage()
        finally:
            store.close()
            if old_sigterm is not None:
                signal.signal(signal.SIGTERM, old_sigterm)
    delta = {key: total_usage[key] - baseline[key] for key in total_usage}
    report = dict(benchmark=benchmark, output=str(stem.with_suffix(".csv")),
                  interrupted=interrupted, this_run=delta, cumulative=total_usage,
                  incomplete_selected=int(((exported.status != "complete")
                                           & exported.item_id.isin(selected_ids)).sum()),
                  approximate_usd=estimate_cost(total_usage, input_price, output_price, cached_input_price))
    print(json.dumps(report, indent=2), flush=True)
    exported.attrs["run_report"] = report
    if interrupted:
        raise KeyboardInterrupt
    return exported


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--benchmark", choices=tuple(BENCHMARKS))
    selection.add_argument("--all", action="store_true", help="All five active notebook benchmarks; MHS is excluded.")
    selection.add_argument("--default-benchmarks", action="store_true", help="The notebook's five active RUN benchmarks.")
    selection.add_argument("--list", action="store_true", help="List labels without API access.")
    parser.add_argument("--limit", type=int, help="First N source rows per benchmark (before resume filtering).")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--checkpoint-every", type=int, default=5, help="CSV export interval; every API response is durable immediately.")
    parser.add_argument("--min-valid-mass", type=float, default=0.90)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--input-price", type=float, default=INPUT_PRICE, help="Estimated USD/million input tokens.")
    parser.add_argument("--cached-input-price", type=float, default=CACHED_INPUT_PRICE)
    parser.add_argument("--output-price", type=float, default=OUTPUT_PRICE)
    args = vars(parser.parse_args(argv))
    if args.pop("list"):
        for name, spec in BENCHMARKS.items():
            print(f"{name}: {spec['levels']}")
        return 0
    all_benchmarks = args.pop("all")
    defaults = args.pop("default_benchmarks")
    benchmark = args.pop("benchmark")
    names = tuple(BENCHMARKS) if all_benchmarks else DEFAULT_BENCHMARKS if defaults else (benchmark,)
    try:
        load_environment()
        incomplete = False
        for name in names:
            result = run_benchmark(name, **args)
            incomplete |= result.attrs["run_report"]["incomplete_selected"] > 0
        return 2 if incomplete else 0
    except KeyboardInterrupt:
        return 130
    except CallFailure as exc:
        print(f"API run stopped: {exc}")
        return 2
    except (RuntimeError, ValueError) as exc:
        print(str(exc))
        return 2
    except Exception as exc:
        # Do not echo arbitrary SDK/network exceptions, headers, or environment.
        print(f"Run stopped ({type(exc).__name__}); received responses remain checkpointed.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
