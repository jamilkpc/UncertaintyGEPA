#!/usr/bin/env bash
# Roda benchmarks com o mestre usando 8000 tokens de raciocinio, no servidor vLLM ja em pe.
#   bash scripts/run_think8k.sh ArMIS HSBrexit
# O nome do experimento sai da configuracao (ex.: gpt56luna__qwen3.5-9b__think8k), entao todos os
# benchmarks desta configuracao ficam na mesma pasta e as tabelas .tex acumulam. Variaveis opcionais:
# LLM1_TAG (padrao gpt56luna), VLLM_URL, WORKERS (padrao 16).
set -euo pipefail
[ "$#" -ge 1 ] || { echo "uso: bash scripts/run_think8k.sh BENCHMARK [BENCHMARK ...]"; exit 1; }
cd "$(dirname "$0")/.."
export VLLM_URL="${VLLM_URL:-http://127.0.0.1:8100/v1/chat/completions}"
export MASTER_THINKING_BUDGET=8000 MASTER_MAX_TOKENS=12000 MASTER_TIMEOUT=1200
exec nice -n 10 python -u scripts/run_pipeline.py --llm1-tag "${LLM1_TAG:-gpt56luna}" \
    --benchmarks "$@" --conditions blind labelled --backend vllm --workers "${WORKERS:-16}"
