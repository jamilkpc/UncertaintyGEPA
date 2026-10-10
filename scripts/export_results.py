#!/usr/bin/env python3
"""Exporta os resultados de experimentos para `experiments/` (versionavel no git), sem o texto dos itens.

    python scripts/export_results.py gpt56luna__qwen3.5-9b__think4k gpt56luna__qwen3.5-9b__think8k

`paper_materials/` fica fora do git porque contem texto dos datasets (tweets, falas abusivas) e respostas brutas
do assessor que os citam. Aqui vai so o que reproduz os numeros e as tabelas:
  metrics/    summary, paired, data, floor e per_item SEM as colunas `text` e `raw`
  tables/     os .tex
  codebooks/  os prompts otimizados
  master_calls.jsonl, experiment_config.json
Tambem copia `paper_materials/comparison/` se existir. Pode rodar de novo: sobrescreve com o estado atual."""
import argparse
import glob
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C      # noqa: E402

DROP = ["text", "raw"]


def export(name, dest_root):
    src = C.OUTDIR / name
    if not src.is_dir():
        raise SystemExit(f"experimento nao encontrado: {src}")
    dst = dest_root / name
    for sub in ("metrics", "tables", "codebooks"):
        shutil.rmtree(dst / sub, ignore_errors=True); (dst / sub).mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(f"{src}/results/*.csv")):
        base = os.path.basename(p)
        if base.startswith(("per_item_", "dev_per_item_")):
            d = pd.read_csv(p); d = d.drop(columns=[c for c in DROP if c in d.columns])
            d.to_csv(dst / "metrics" / base, index=False)
        elif base.startswith("floor_") and not base.startswith("floor_summary_"):
            d = pd.read_csv(p); d.drop(columns=[c for c in DROP if c in d.columns]).to_csv(dst / "metrics" / base, index=False)
        else:
            shutil.copy(p, dst / "metrics" / base)
        n += 1
    for p in glob.glob(f"{src}/paper/*.tex"): shutil.copy(p, dst / "tables" / os.path.basename(p)); n += 1
    done = {os.path.basename(p)[len("summary_"):-4] for p in glob.glob(f"{src}/results/summary_*.csv")}
    for p in glob.glob(f"{src}/work/codebook_*.txt"):      # so benchmarks com resultado (nada de run incompleto)
        if os.path.basename(p)[len("codebook_"):].split("__")[0] in done:
            shutil.copy(p, dst / "codebooks" / os.path.basename(p)); n += 1
    for f in ("experiment_config.json", "work/master_calls.jsonl"):
        if (src / f).exists(): shutil.copy(src / f, dst / os.path.basename(f)); n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+")
    a = ap.parse_args(argv)
    root = C.ROOT / "experiments"; root.mkdir(exist_ok=True)
    for e in a.experiments:
        print(f"{e}: {export(e, root)} arquivos -> experiments/{e}/")
    comp = C.OUTDIR / "comparison"
    if comp.is_dir():
        shutil.rmtree(root / "comparison", ignore_errors=True); shutil.copytree(comp, root / "comparison")
        print("comparison/ copiado")


if __name__ == "__main__":
    main()
