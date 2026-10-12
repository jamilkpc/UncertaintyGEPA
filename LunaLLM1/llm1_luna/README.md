# Standalone LLM1 with GPT-5.6 Luna

This package extracts only LLM1 from `10_all_benchmarks_v2.ipynb`. It never executes
or modifies that notebook, its other models, or its Gemma caches. The five active
notebook benchmarks are supported; **MHS is excluded at the user's request**.

| CLI benchmark | LLM1 task | Valid labels |
|---|---|---|
| `HSBrexit` | Hate speech in Brexit/immigration tweets | 0 = no, 1 = yes |
| `ArMIS` | Misogyny/sexism in Arabic tweets | 0 = no, 1 = yes |
| `ConvAbuse` | Abuse in the final user turn of a chatbot conversation | 0 = not abusive, 1 = abusive |
| `MDAgreement` | Offensiveness in English tweets | 0 = no, 1 = yes |
| `AmbiStory` | Plausibility of a candidate meaning in a narrative | 1 = completely implausible through 5 = clearly intended |

The ConvAbuse source's human coding uses 1 for **not abusive**. Its notebook LLM1
prompt uses 1 for **abusive**; this package preserves the LLM1 coding.

## Source fidelity

`benchmarks.py` contains the exact prompts, label sets, item IDs, text formatting,
and preprocessing from notebook cells 5–6 (zero based), excluding MHS. The loader
only adds a 60-second URL timeout. Constants from cell 3 are `R=20`,
`TEMP_RESAMPLE=1.0`, and `SEED=42`. The notebook's cell 8 uses population variance,
so this package also uses `ddof=0`, not sample variance.

Data locations are preserved:

- HSBrexit, ArMIS, ConvAbuse, MDAgreement:
  `https://raw.githubusercontent.com/Le-Wi-Di/le-wi-di.github.io/main/LeWiDi_2-2023/{d}_dataset/{d}_{split}.json`
  where `d` is respectively `HS-Brexit`, `ArMIS`, `ConvAbuse`, or `MD-Agreement`.
- AmbiStory: `https://raw.githubusercontent.com/Janosch-Gehring/ambistory/main/{split}.json`.
- Each loader reads train, dev, and test in that order. IDs retain their original
  `{dataset}-{split}-{source_key}` spelling. ConvAbuse preserves dialogue context;
  AmbiStory preserves narrative, ambiguous word, and candidate meaning. Invalid
  binary human annotations are filtered exactly as in the original loader.

The first load saves a separate JSON snapshot of the full preprocessed corpus.
`--limit` selects its first N rows **before** completed items are removed. Repeating
`--limit 3` therefore never advances to another three items. Dataset snapshots
preserve input order/text across resumes and can also supply the downstream merge.

## Environment and installation

Use Python 3.12+ on macOS/Linux (validated with Python 3.13). From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r llm1_luna/requirements.txt
```

Select `.venv/bin/python` as the interpreter for `11_generate_llm1_luna.ipynb`.
The installed versions used for validation are recorded in `validation.json`.
Runtime dependencies are `openai`, `python-dotenv`, `numpy`, and `pandas`.
`nbformat` and `ipykernel` support the notebook; tests use standard-library unittest.
No GEPA or downstream model dependencies are required.

The package reads the existing **project-root** `.env` using `python-dotenv`:

```python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(PROJECT_ROOT / ".env")
# Check presence and fail before downloading data or starting API work.
client = OpenAI()
```

The explicit path is equivalent to `load_dotenv()` from this project, and remains
correct when another working directory launches the package. Existing environment
variables take precedence. The SDK reads the key from the environment; no literal
credential is passed to the client. A missing/empty key raises:

```text
OPENAI_API_KEY not found. Add it to the project's .env file.
```

`.env` is ignored by Git and is never generated or overwritten. `.env.example`
contains only the empty `OPENAI_API_KEY=` assignment. Credential checks print only
a boolean. API exception bodies/headers are not printed or stored.

## Commands

List the five supported benchmarks without API access:

```bash
python -m llm1_luna.run_llm1 --list
```

Run a five-item test (up to 105 successful responses for previously unseen items):

```bash
python -m llm1_luna.run_llm1 --benchmark HSBrexit --limit 5 --workers 1
```

Run one full benchmark, or all five, when intentionally ready:

```bash
python -m llm1_luna.run_llm1 --benchmark HSBrexit --workers 4
python -m llm1_luna.run_llm1 --all --workers 4
```

`--default-benchmarks` is an alias for the notebook's active list, which is now the
same five benchmarks as `--all`. Default concurrency is **one in-flight request**;
`--workers 4` permits at most four, with no nested request pools. Omitting the
required benchmark selector prints help/error and never starts a run. The notebook
has three-item smoke tests and a disabled switch around the full-run cell.

Other options: `--output-dir`, `--checkpoint-every` (CSV exports, default 5 items),
`--min-valid-mass` (default 0.90), `--max-attempts` (default 5), `--timeout` (seconds,
default 60), and `--input-price`, `--cached-input-price`, `--output-price` (USD per
million tokens). Resample count, temperatures, and model are fixed to the requested
experiment, and changing scientific settings requires a new output directory.

## Calls and uncertainty measures

Each successful item has **21 response slots**:

1. Slot 0: canonical label at temperature 0.
2. Slots 1–20: twenty resamples at temperature 1. **Slot 1 also requests
   `top_logprobs=20` and `include=["message.output_text.logprobs"]`.** It is counted
   in the twenty draws; there is no extra logprob request.

All requests use Responses, `model="gpt-5.6-luna"`,
`reasoning={"effort": "none"}`, `store=False`, and the original system prompt plus
`ITEM:\n{text}`. There is no model substitution, extra label alphabet, JSON output
constraint, or changed benchmark instruction. Responses requires a minimum
`max_output_tokens` of 16, so the package uses 16 rather than the original Gemma
cap of 4; the prompt still asks for one character. Unsupported request settings
fail visibly rather than silently changing the experiment.

The parser accepts only a complete valid label after stripping whitespace. It
intentionally rejects explanations, decimals, multiple labels, refusals, or
truncated responses. This is stricter than the original first-digit regex.
Malformed stochastic outputs remain `null` in the 20-draw vector and are excluded
from variance; they are not replaced with additional draws. A single valid draw
has variance zero, and no valid draws produce a missing variance.

`resampling_var = mean((draw - mean(valid_draws))**2)` over valid draws only.

`logprob_var` uses **the first output-text token position**. For each top-20 token
whose stripped text equals a valid label, it adds `exp(logprob)` to that label's
mass. Distinct variants such as `1` and ` 1` are summed, but duplicate identical
tokens are not counted twice. It records the summed `valid_label_prob_mass` before
normalizing over observed valid-label mass, then computes:

```text
p[k] = observed_mass[k] / valid_label_prob_mass
mu = sum(p[k] * k)
logprob_var = sum(p[k] * (k - mu)**2)
```

Top-20 output is truncated information: absent labels have zero **observed** mass,
not proven zero true probability. Missing labels and the original candidate tokens
are retained in SQLite diagnostics. No smoothing or uniform fallback is used.
If mass is below the configurable 0.90 floor, no labels appear, the API omits
logprobs, or the probabilities are invalid, `logprob_var` is NaN and
`label_probabilities` is `{}`. The mass and reason are preserved. The implementation
does not skip a leading whitespace/rationale token to condition on a later digit.

When enough mass is present but some labels are absent, the explicit
`logprob_status=ok_partial_label_support` flags that the variance uses only the
observed support. `ok` means every benchmark label appeared. Even a request for
20 candidates can return fewer; the package records the actual returned count.

A completed item with unusable logprobs remains completed with a diagnostic NaN;
resuming does not charge it again to fish for a better distribution.

Official references: [Luna model and prices](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
and [Responses request/response fields](https://developers.openai.com/api/reference/java/resources/beta/subresources/responses/methods/create).

## Output and downstream compatibility

Files have the separate prefix `work/llm1_gpt56luna_<benchmark>`:

- `.csv`: per-item export, including status and incomplete-item diagnostics.
- `.sqlite3`: durable request ledger, response tokens/logprobs, per-item summaries,
  and configuration. Keep it with the CSV to resume partial items.
- `.metadata.json`: model/configuration, prompt once, prompt/source hashes, original
  notebook hash, SDK/data-library versions, timestamps, usage and price assumptions.
- `.lock`: an operating-system writer lock, automatically released on process exit.
- `work/llm1_gpt56luna_inputs_<benchmark>.json`: full preprocessed input snapshot.

The **exact CSV column order** (output schema version 2) is defined once in
`checkpoint.OUTPUT_COLUMNS`:

```text
item_id, benchmark, llm1_label, human_label, human_uncertainty,
n_human_annotations, resampling_var, logprob_var,
valid_label_prob_mass, n_valid_resamples, status, resampling_draws, label_probabilities,
logprob_status, input_tokens, output_tokens, cached_input_tokens,
total_requests, successful_responses, n_resamples_completed,
model, created_at, updated_at, prompt_hash, input_hash, error
```

Labels/counts are integers when present; CSV readers may infer float labels if
incomplete rows contain blanks. Variances and mass are floats; missing numeric
values are empty CSV fields (`NaN` in pandas). `resampling_draws` and
`label_probabilities` are standards-compliant JSON strings (invalid draws are JSON
`null`). Probabilities' JSON keys are label strings. IDs and hashes are strings;
timestamps use UTC ISO 8601. `status` is `complete`, `incomplete`, `api_error`, or
`malformed_output`. `logprob_status` explains missing/low-mass logprobs.

The main variables have distinct meanings:

| Column | Definition / source |
|---|---|
| `llm1_label` | Luna's canonical annotation at temperature 0, in the benchmark's original label scale. |
| `human_label` | The original loader's `consensus`, copied from the frozen input snapshot. |
| `human_uncertainty` | The original loader's `u_m`, copied from the same snapshot. Continuous human disagreement, not an LLM prediction or a discrete uncertainty class. |
| `n_human_annotations` | The original loader's `n_ann`. |
| `resampling_var` | Population variance of valid numeric labels among the 20 temperature-1 responses. |
| `n_valid_resamples` | Number of valid labels used in `resampling_var`. |
| `logprob_var` | Variance under the normalized observed valid-label token probabilities. |
| `valid_label_prob_mass` | Valid-label probability mass before normalization. |

For the four binary benchmarks, human uncertainty is `p * (1 - p)`, where `p`
is the fraction of human annotations equal to 1 (range 0–0.25). HSBrexit, ArMIS,
and MDAgreement use `int(p > 0.5)` for consensus; ties produce 0. ConvAbuse's raw
1 means nonabusive, so its original loader uses `int(p <= 0.5)` to put the human
consensus on the LLM scale (1 = abusive; ties produce 1). Its variance is unchanged
by this reversal. AmbiStory uses population variance of the human ratings on 1–5
(range 0–4), with `int(round(mean(ratings)))` as its human label, including Python's
ties-to-even rounding. These definitions are inherited unchanged from the source
notebook; human variances are not rescaled across benchmarks.

The redundant CSV aliases `label`, `g_m`, and `n_valid_draws` are removed. New item
results are constructed with their final names in `run_llm1._item_result`; there
is no chain of downstream renames. A single compatibility reader,
`checkpoint.read_result`, accepts older checkpoints, checks that duplicate values
agree, and drops their aliases. It never changes the raw response ledger. Output
schema versioning is separate from the unchanged sampling configuration.

Read completed results directly and merge by `item_id` within each benchmark:

```python
frame = load_benchmark("HSBrexit")
llm1 = pd.read_csv("work/llm1_gpt56luna_HSBrexit.csv", dtype={"item_id": str})
llm1 = llm1.loc[llm1.status == "complete"]
merged = frame.merge(llm1, on="item_id", validate="one_to_one")
```

The original downstream notebook expects `g_m` and `n_valid_draws`. If that loader
is later adapted, select `resampling_var` and `n_valid_resamples` in their place;
`llm1_label` is unchanged. No LLM2/LLM3 code is modified here.

To upgrade an already completed Luna benchmark without creating an API client:

```bash
python -m llm1_luna.export_results --benchmark HSBrexit --benchmark ArMIS \
  --output-dir work/llm1_gpt56luna_paper
```

This optional command uses the same export schema as the normal runner. It checks
exact item IDs and source-text hashes, copies human aggregates from the frozen
snapshot, and audits all saved draws/logprobs against the raw responses. It refuses
active writers or incomplete benchmarks. Its `.export_audit.json` records source
and output hashes, row counts, and identical API usage before and after export.
Future runs and resumes include the human columns automatically. No responses are
re-sampled to change the output schema.

No text/prompt is repeated in result rows; source text resides once in the input
snapshot. Existing Gemma filenames `work/llm1_<benchmark>.csv` are never read or written.

## Resumption, errors, and cost accounting

SQLite commits each received response immediately with `synchronous=FULL`. Results
and metadata export atomically every five completed/failed items and on normal exit
or handled interruption. Restart with the same command: completed IDs are skipped;
partial items reuse their received responses and fill only missing slots. A stale
CSV after abrupt termination is rebuilt from the ledger. CSV-only recovery also
skips completed rows if matching metadata is present, but cannot recover partial
draws without the database. A changed prompt, setting, or existing item's text is
rejected to prevent mixing experiments. One process may write each benchmark.

Ctrl-C/SIGTERM stops scheduling and saves in-flight responses before exit. A second
forced termination, machine failure, or a timeout after server processing can occur
between a charged response and its durable receipt. Exactly-once billing in that
unobservable interval cannot be guaranteed; recorded completed responses are reused.

Connection/timeouts, 408/409/429 and 5xx errors retry with bounded exponential
backoff and jitter, respecting numeric Retry-After up to 60 seconds. SDK automatic
retries are disabled to avoid multiplying attempts. Authentication/permission,
invalid model/parameters, and quota failures stop the run; the model and sampling
settings are not silently changed. An exhausted transient error leaves the item
retryable on the next invocation. Malformed canonical labels can be retried on the
next invocation; received stochastic draws are retained. If **all** stochastic
draws are malformed, the row stays `malformed_output`; investigate before explicitly
starting a new experiment in a new output directory.

Usage totals include input, cached input, output, attempts, and successful responses,
including any earlier invalid canonical response. `total_requests` counts recorded
attempt starts, including retries and interrupted attempts; API usage is known only
for received responses. Ordinary complete items have 21 attempts and responses.
Resumption prints zero new requests for already completed items. Each benchmark
reports cumulative usage; `--all` prints one cumulative report per benchmark.

Approximate cost uses centralized configurable prices (initially $0.20 input,
$0.02 cached input, and $1.20 output per million tokens). These are estimates based
on returned token counts, not actual billing; special service tiers, cache writes,
long-context multipliers, unreported failed-call usage, and later price changes may
differ. Temperature zero is not a promise of exact deterministic API outputs.

## Validation

Offline behavioral tests and an audit of saved real outputs:

```bash
python -m unittest discover -s llm1_luna/tests -v
python -m llm1_luna.verify_outputs --benchmark HSBrexit
python -m llm1_luna.verify_outputs --benchmark AmbiStory
```

The tests check source-registry fidelity, parsing, exponentiation, token variants,
normalization, low/missing mass, no shifting to a later token, exactly 20 resamples
including one logprob call, malformed draws, partial-item interruption, completed
item resume, old/new CSV recovery, human-field joins by item ID, offline schema
upgrades without API calls, and protection against changed input. `validation.json`
records the live smoke-test results and empirical response-layout observations.

## Observed smoke-test behavior

The validation run completed four HSBrexit and three AmbiStory items, with exactly
21 successful response slots per item (147 requests). A raw HTTP probe was saved
as the fourth HSBrexit item's first resample and reused when completing that item.
The outgoing request and the returned response both contained `top_logprobs=20`;
the SDK's parsed output exactly matched the raw JSON. Nevertheless, the API returned
only one candidate for the HSBrexit items and two or three for the AmbiStory items.
This is observed API behavior, not candidate loss in the parser.

For `AmbiStory-train-0`, the returned label logprobs were 5: -0.484123,
4: -1.030060, and 3: -4.186337. Their valid-label mass was 0.9884248083; normalized
probabilities were respectively 0.6234540, 0.3611661, and 0.0153799, yielding
`logprob_var=0.2690797727`. The independent 20-draw variance was 0.8475.
The missing labels remain flagged as partial support.

The API reported five output tokens for a one-digit answer, with zero reasoning
tokens in the inspected raw response. Cost accounting uses returned usage, not
visible character counts. Repeating the smoke commands made zero new requests.
The run used 20,937 input and 735 output tokens (no cached input), an approximate
$0.00507 at the configured prices. No full benchmark was launched.

## Fresh paper run (separate from smoke tests)

The full paper run uses `work/llm1_gpt56luna_paper/`. It starts every annotation
fresh and imports only the source dataset snapshots. Smoke-test CSVs and ledgers
remain outside that directory and never contribute annotations to the paper data.
There are 20,664 items: HSBrexit 1,120; ArMIS 943; ConvAbuse 4,050;
MDAgreement 10,753; AmbiStory 3,798. Without retries, the run requires 433,944
responses (21 per item). MHS remains excluded.

Launch or resume the separate paper run with:

```bash
python -m llm1_luna.paper_run --workers 4 --background
```

The supervisor writes `paper_run.log`, updates `paper_run_status.json` every
30 seconds, and saves the code/data fingerprints in `paper_run_manifest.json`.
It processes the five benchmarks, makes at most three passes to recover incomplete
items, and numerically audits every completed result. The final status records
coverage, missing logprob variances, usage, and approximate cost. `complete` means
all items were generated and audited; the report still explicitly counts any
logprob NaNs required by the original low-mass rule. `needs_review` identifies
remaining incomplete items, and `failed` records a sanitized fatal error.

Every response is still checkpointed immediately; this long run exports CSVs
every 25 items. On macOS, the running supervisor prevents automatic idle sleep
while it is active. The machine and external drive must remain available.
Interrupt with `kill -TERM <pid>` using the PID in `paper_run_launch.json`, then
resume with the same command. Completed paper items are reused on resumption;
smoke-test annotations are never imported.

Native Python faults (for example, `SIGBUS`) can terminate the process before it
updates the status JSON. A stale `running` status is therefore not proof of a live
process: check the recorded PID and recent log activity. Native-fault logging is
enabled with `faulthandler` so thread stack locations go to `paper_run.log`, without
local variable values. Resume with the same command after the writer has exited;
the checkpoint reuses every saved response. Requests in flight without a saved
response may need repeating; their server-side completion and billing are unknown.

## Numeric precision and storage recovery (2026-10-02)

The full run exposed tiny API numeric errors: 34 valid-label logprobs of +0.000004
and observed valid-label masses exceeding one by at most 0.000006477 among the
first 891 completed items. The extractor now explicitly tolerates 0.00001 of
numerical error. It preserves the original `exp(logprob)` values and their raw
sum (which can therefore be slightly above one), then normalizes as usual. It
does not clamp or invent label probabilities. Such rows have a logprob status
ending in `_numeric_roundoff`. Larger violations remain invalid. All existing
results can be recalculated from saved raw responses without additional API calls.
This precision issue is separate from incomplete top-20 label support.

After a local SQLite OperationalError, the checkpoint passed `PRAGMA quick_check`.
The original error log retained only the exception type, so the precise failure
message was unavailable. Storage recovery removes competing SQLite reads from
the heartbeat: progress now comes from atomic export metadata and shows its
`as_of` timestamp. The writer tolerates a temporary lock for up to 60 seconds;
input registration and recalculated summaries are committed in batches. Counters
advance only after successful commits. Future SQLite failures record their error
name and local error text. Source fingerprints for every resumed launch are kept
in `paper_run_launch_history.json`; the initial manifest remains intact.
