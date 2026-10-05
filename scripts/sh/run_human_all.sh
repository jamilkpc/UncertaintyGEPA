#!/usr/bin/env bash
# Versao "humano": a LLM 1 e trocada pelos anotadores humanos (rotulo = consenso, alvo = desacordo humano u_m); o resto
# do pipeline e igual. Roda os benchmarks EM SEQUENCIA, um por vez, no servidor vLLM ja em pe, com o mestre usando
# 8000 tokens de raciocinio, SO na condicao blind (CONDITIONS="blind labelled" para incluir a labelled). Resultados em paper_materials/human__<modelo>__think8k/ (nao mexe nos outros experimentos).
#
#   bash scripts/sh/run_human_all.sh                      # todos, do menor para o maior
#   bash scripts/sh/run_human_all.sh ArMIS ConvAbuse      # so estes, nesta ordem
#   NOWAIT=1 bash scripts/sh/run_human_all.sh ...         # nao espera outros runs terminarem
#
# Por padrao espera qualquer run_pipeline em andamento acabar (para nao dividir a GPU). Se um benchmark falhar, segue
# para o proximo. Cada benchmark e retomavel: rodar de novo pula o que ja existe.
set -uo pipefail
cd "$(dirname "$0")/../.."
BENCHES=("$@"); [ "${#BENCHES[@]}" -gt 0 ] || BENCHES=(ArMIS HSBrexit AmbiStory ConvAbuse MDAgreement)
export CONDITIONS="${CONDITIONS:-blind}"

if [ -z "${NOWAIT:-}" ]; then
  while pgrep -f "scripts/run_pipeline.py" > /dev/null; do sleep 60; done
fi
for b in "${BENCHES[@]}"; do
  echo ">>> $b  ($(date '+%d/%m %H:%M'))"
  LLM1_TAG=human bash scripts/sh/run_think8k.sh "$b" || echo "!!! $b falhou; seguindo para o proximo"
done
echo ">>> fim  ($(date '+%d/%m %H:%M'))"
