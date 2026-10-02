# UncertaintyGEPA

Pipeline that makes an LLM state its uncertainty about an annotation in words, and tunes the prompt that
guides it with [GEPA](https://github.com/gepa-ai/gepa) so the stated uncertainty tracks how unstable a
second LLM's labels are across resamples.

## How it works

```
 LLM 1 results  ─┐   (label + 20 resampled labels per item)
                 ├─►  frames  ─►  GEPA search  ─►  optimised prompts  ─►  test evaluation  ─►  summary_*.csv  ─►  .tex
 dataset loader ─┘   (text, human disagreement, split)    train + dev                              test
```

**Roles**

| Role | Does | Output |
|---|---|---|
| **LLM 1, annotator** | labels each item at temperature 0, then is resampled 20 times at temperature 1 | `llm1_label`, and `g_m`, the variance of the 20 draws |
| **LLM 2, assessor** | reads the item (and optionally the label) and answers with one hedge, no reasoning | `not uncertain`, `somewhat uncertain` or `highly uncertain` |
| **LLM 3, master** | writes and evolves the assessor's prompt with GEPA, with reasoning | one optimised prompt per benchmark and condition |

**Conditions.** `blind`: the assessor sees the text only. `labelled`: it sees the text and the LLM 1 label.

**Splits.** `train` is split into folds that GEPA scores; `dev` selects candidates; `test` is used once, at the end.

**Search target.** GEPA maximises the Spearman correlation between the hedge and `g_m`. Human disagreement `u_m`
and the consensus label are never used in the search; they only appear in the evaluation.

**Evaluation.** Each method is scored on the test split against both `g_m` and `u_m`. Methods: `single_hedge`
(the unoptimised seed prompt), `reclassify` (labelled only), `topk_wrong`, `pros_cons`, and `GEPA`.

## Repository structure

| Path | Purpose |
|---|---|
| `10_all_benchmarks.ipynb` | original all-in-one notebook, kept as is |
| `scripts/config.py` | backend, models, constants, output folders |
| `scripts/benchmarks.py` | benchmark registry and dataset loaders |
| `scripts/llm.py` | HTTP calls; the annotator, assessor and master |
| `scripts/metrics.py` | Spearman, bootstrap, folds, scale ceiling |
| `scripts/prompts.py` | seed prompt, baselines, GEPA reflection template |
| `scripts/gepa_search.py` | GEPA adapter and search |
| `scripts/pipeline.py` | loads corpora, estimates cost, builds the frames |
| `scripts/evaluation.py` | test evaluation, noise test, noise floor |
| `scripts/paper.py` | writes the `.tex` tables and macros |
| `scripts/run_pipeline.py` | command-line runner |
| `scripts/run.ipynb` | notebook to run stages and inspect results |
| `scripts/README.md` | running on a GPU server |
| `paper_materials/` | inputs and outputs, git-ignored (see Outputs) |

## Benchmarks

Downloaded on first use. Default run: the first six; edit `RUN` in `scripts/benchmarks.py` or pass `--benchmarks`.

| Key | Dataset | Construct |
|---|---|---|
| `HSBrexit`, `HSBrexitOff` | LeWiDi 2023, HS-Brexit | hate speech; offensive language |
| `ArMIS` | LeWiDi 2023 | misogyny and sexism (Arabic) |
| `ConvAbuse` | LeWiDi 2023 | abuse toward a chatbot |
| `MDAgreement` | LeWiDi 2023 | offensiveness |
| `AmbiStory` | SemEval-2026 Task 5 | plausibility of a word meaning in a story |
| `MHShatespeech`, `MHSdehumanize`, `MHSattackdefend` | Measuring Hate Speech | not in the default run |

Each item carries the text, the human disagreement `u_m`, the consensus label and the published split.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r scripts/requirements.txt     # gepa is pinned to 0.1.4
```

Pick a backend with `BACKEND` (or `--backend`):

| Backend | Needs |
|---|---|
| `ollama` | Ollama running; model in `OLLAMA_MODEL` (default `qwen3.5:9b`) |
| `vllm` | a running `vllm serve`; `VLLM_URL`, `VLLM_MODEL`, `VLLM_WORKERS` |
| `openrouter` | `OPEN_ROUTER_API_KEY`; `OR_ANNOTATOR`, `OR_ASSESSOR`, `OR_MASTER` |
| `cloudflare` | secrets `CF_Palver_1`, `CF_Palver_2` |

Secrets come from environment variables (or Colab `userdata`). Never commit keys.

## Run

LLM 1 results: either let the pipeline generate them (omit `--llm1-tag`), or reuse existing ones by placing
`llm1_<tag>_<benchmark>.csv` in `paper_materials/inputs/llm1/` and passing `--llm1-tag <tag>`. Columns used:
`item_id`, `llm1_label`, `resampling_var`, `n_valid_resamples`, `status` (rows not `complete` are dropped).

```bash
# estimate cost; loads data, calls no model
python scripts/run_pipeline.py --dry-run --experiment myrun --llm1-tag mytag --benchmarks ConvAbuse

# full run
python -u scripts/run_pipeline.py --experiment myrun --llm1-tag mytag \
    --benchmarks ConvAbuse --conditions blind labelled --backend vllm
```

| `--stages` | Does |
|---|---|
| `gepa` | optimised prompt per benchmark and condition |
| `eval` | scores every method on the test split |
| `floor` | repeats the evaluation `--floor-k` times to measure run-to-run noise |
| `paper` | writes the `.tex` tables and macros |
| `noise` | quick noise test on the seed prompt |

Default stages: `gepa eval paper`. A quick assessor and master check runs first. Runs are resumable: rerun the
same command and finished work is skipped. Design parameters (folds, GEPA budget) are in `scripts/config.py`.

## Outputs

```
paper_materials/<experiment>/
  work/      codebook_<bench>__<condition>.txt, trajectory_*.json, gepa_runs/
  results/   summary_<bench>.csv, per_item_<bench>.csv, floor_*.csv
  paper/     tab_*.tex, macros.tex, codebooks.tex
  logs/      one log per run
```
Each experiment has its own folder, so a new run never reuses another run's prompts or overwrites its results.
Everything under `paper_materials/` is git-ignored: copy results out of the machine you ran on.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `CERTIFICATE_VERIFY_FAILED` on download | Python without system certificates; `certifi` is used if installed |
| `content vazio` | reasoning used the token budget; the master falls back to the last `---` block of its reasoning |
| `servidor nao responde` | start `vllm serve` / Ollama, and open the SSH tunnel if remote |
| a run died | rerun the same command |
