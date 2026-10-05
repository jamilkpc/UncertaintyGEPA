#!/usr/bin/env bash
# Ablation de custo: o alvo de instabilidade da LLM 1 passa a ser a variancia dos LOGPROBS (1 chamada por item) em vez da
# variancia das R=20 amostras (21 chamadas). O rotulo da LLM 1 e o resto do pipeline sao os mesmos. Roda os benchmarks EM
# SEQUENCIA, so na condicao blind, com o mestre usando 8000 tokens de raciocinio. Resultados em
# paper_materials/gpt56luna-logprob__<modelo>__think8k/ (outro experimento; nao mexe nos demais).
#
#   bash scripts/sh/run_logprob_all.sh                    # os 5 benchmarks, do menor para o maior
#   bash scripts/sh/run_logprob_all.sh ArMIS ConvAbuse    # so estes, nesta ordem
#   NOWAIT=1 bash scripts/sh/run_logprob_all.sh ...       # nao espera a fila da LLM 1 terminar
#
# Por padrao espera a fila da versao com LLM 1 (--llm1-tag gpt56luna, no servidor da GPU 1) acabar. Exige 3 checagens
# vazias seguidas (30 s), porque entre um benchmark e o seguinte da fila ha um instante sem processo. Usa o servidor
# vLLM da porta 8100 (VLLM_URL para mudar). Cada benchmark e retomavel.
set -uo pipefail
cd "$(dirname "$0")/../.."
BENCHES=("$@"); [ "${#BENCHES[@]}" -gt 0 ] || BENCHES=(ArMIS HSBrexit AmbiStory ConvAbuse MDAgreement)
export CONDITIONS="${CONDITIONS:-blind}" TARGET=logprob LLM1_TAG="${LLM1_TAG:-gpt56luna}"

if [ -z "${NOWAIT:-}" ]; then
  empty=0
  while [ "$empty" -lt 3 ]; do
    if pgrep -f "llm1-tag ${LLM1_TAG} " > /dev/null; then empty=0; else empty=$((empty + 1)); fi
    sleep 30
  done
fi
for b in "${BENCHES[@]}"; do
  echo ">>> $b  ($(date '+%d/%m %H:%M'))"
  bash scripts/sh/run_think8k.sh "$b" || echo "!!! $b falhou; seguindo para o proximo"
done
echo ">>> fim  ($(date '+%d/%m %H:%M'))"
