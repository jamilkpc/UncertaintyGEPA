#!/usr/bin/env python3
"""Compara experimentos (configuracoes) lado a lado, sem chamar nenhuma LLM.

    python scripts/compare_experiments.py gpt56luna__qwen3.5-9b__think4k gpt56luna__qwen3.5-9b__think8k

Le `paper_materials/<experimento>/results/summary_<bench>.csv` e grava em `paper_materials/comparison/`:
`summary_all.csv` (tudo, com uma coluna por experimento) e `tab_experiments.tex` (rho(h,u) por benchmark,
condicao e metodo em cada experimento, e a amplitude entre eles). Com um prompt fixo (ex.: o hedge unico), a
amplitude e ruido do assessor; no GEPA ela inclui tambem a aleatoriedade da busca."""
import argparse
import glob
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C                      # noqa: E402
from benchmarks import RUN              # noqa: E402

METHOD_ORDER = ["single_hedge", "reclassify", "topk_wrong", "pros_cons", "GEPA"]


def esc(s): return str(s).replace("_", r"\_")
def fmt(x, d=3): return "--" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{d}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", help="nomes das pastas em paper_materials/")
    a = ap.parse_args(argv)
    frames = []
    for e in a.experiments:
        parts = sorted(glob.glob(str(C.OUTDIR / e / "results" / "summary_*.csv")))
        if not parts:
            raise SystemExit(f"sem resultados em {C.OUTDIR / e}/results")
        d = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True); d["experiment"] = e; frames.append(d)
    allr = pd.concat(frames, ignore_index=True)
    # rotulos curtos: a parte do nome que difere entre os experimentos
    segs = [e.split("__") for e in a.experiments]
    label = {e: (s[-1] if len({x[-1] for x in segs}) == len(segs) else e) for e, s in zip(a.experiments, segs)}
    allr["label"] = allr.experiment.map(label)
    out = C.OUTDIR / "comparison"; out.mkdir(exist_ok=True)
    allr.to_csv(out / "summary_all.csv", index=False)

    labels = [label[e] for e in a.experiments]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lll" + "r" * len(labels) + "r}", r"\hline",
         "Benchmark & Condition & Method & " + " & ".join(esc(x) for x in labels) + r" & range \\", r"\hline"]
    benchmarks = [b for b in RUN if b in set(allr.benchmark)] + sorted(set(allr.benchmark) - set(RUN))
    for b in benchmarks:
        for cond in ("blind", "labelled"):
            for m in [m for m in METHOD_ORDER if m in set(allr.method)]:
                sub = allr[(allr.benchmark == b) & (allr.condition == cond) & (allr.method == m)]
                if sub.empty: continue
                vals = [sub[sub.label == x].rho_u for x in labels]
                vals = [float(v.iloc[0]) if len(v) else float("nan") for v in vals]
                ok = [v for v in vals if not math.isnan(v)]
                rng = fmt(max(ok) - min(ok)) if len(ok) > 1 else "--"
                L.append(f"{esc(b)} & {cond} & {esc(m)} & " + " & ".join(fmt(v) for v in vals) + f" & {rng} \\\\")
    L += [r"\hline", r"\end{tabular}",
          r"\caption{$\rho(h_m,u_m)$ on the test split in each experiment (a configuration of the master). Range is "
          r"the largest minus the smallest value across experiments: for a fixed prompt it is run-to-run noise of the "
          r"assessor; for the codebook it also includes the randomness of the search.}",
          r"\label{tab:experiments}", r"\end{table}"]
    open(out / "tab_experiments.tex", "w").write("\n".join(L))
    print(f"escritos em {out}: summary_all.csv, tab_experiments.tex  (experimentos: {labels})")
    piv = allr.pivot_table(index=["benchmark", "condition", "method"], columns="label", values="rho_u")
    piv["range"] = piv.max(axis=1) - piv.min(axis=1)
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
