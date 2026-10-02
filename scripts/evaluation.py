"""Avaliacao no teste (todos os metodos) e o teste de ruido do assessor a temperatura 0."""
import collections
import glob
import os

import numpy as np
import pandas as pd

import config as C
import metrics as M
from benchmarks import BENCHMARKS
from llm import assess_blind_meta, run_assessor
from prompts import baselines, seed_codebook


def evaluate_test(frames, codebooks, run=None):
    """Roda cada metodo no split de teste de cada benchmark e condicao.
    Grava `results/summary_<bench>.csv` e `results/per_item_<bench>.csv` (um par por benchmark, para
    rodar outro benchmark depois nao sobrescrever este) e devolve (summary, per_item) do que rodou."""
    rows, per_item = [], []
    for b in (run or list(frames)):
        n_rows, n_items = len(rows), len(per_item)
        f = frames[b]; test = M.test_fold_with_human(f[f.split_orig == "test"])
        K_u, ceil_u = M.rho_max_discrete(test["um"]); K_g, ceil_g = M.rho_max_discrete(test["gm"])
        for cond in C.CONDITIONS:
            meths = dict(baselines(b, cond)); meths["GEPA"] = codebooks[f"{b}__{cond}"]
            for name, prompt in meths.items():
                h, raw = run_assessor(prompt, test, cond, b)
                per_item.append(pd.DataFrame({
                    "benchmark": b, "condition": cond, "method": name,
                    "item_id": test["ids"], "text": test["texts"],
                    "llm1_label": test["labels"], "consensus": test["consensus"],
                    "g_m": test["gm"], "u_m": test["um"],
                    "hedge": [C.HEDGES[x] if isinstance(x, int) else None for x in h], "raw": raw}))
                ru = M.spearman(h, test["um"]); rg = M.spearman(h, test["gm"])
                lo, hi = M.spearman_ci(h, test["um"])
                dist = collections.Counter(x for x in h if x is not None)
                rows.append({
                    "benchmark": b, "condition": cond, "method": name,
                    "rho_u": ru, "ci_lo": lo, "ci_hi": hi, "ceil_u": ceil_u, "pct_ceil": ru / ceil_u,
                    "rho_g": rg, "ceil_g": ceil_g, "coverage": M.coverage(h),
                    "levels_used": len(dist), "modal_share": (max(dist.values()) / sum(dist.values())
                                                              if dist else float("nan")),
                    "n_test": len(test["ids"]), "K_u": K_u})
                print(f"{b:14} {cond:9} {name:14} rho_u={ru:+.3f} [{lo:+.2f},{hi:+.2f}]  "
                      f"rho_g={rg:+.3f}  niveis={len(dist)}  cov={M.coverage(h):.0%}")
        pd.DataFrame(rows[n_rows:]).to_csv(f"{C.RESDIR}/summary_{b}.csv", index=False)
        pd.concat(per_item[n_items:]).to_csv(f"{C.RESDIR}/per_item_{b}.csv", index=False)
    return pd.DataFrame(rows), pd.concat(per_item)


def noise_test(bench="HSBrexitOff", k=3, provider="keep"):
    """Repete `k` vezes o mesmo prompt (condicao blind, itens de teste de `bench`) e mede quanto o
    hedge muda entre execucoes. Nao usa o LLM 1: o assessor so le o texto e u_m vem do loader.

    provider: "keep" usa C.OR_PROVIDER["assessor"]; None deixa o roteador escolher; ou um nome.
    Devolve (runs, summary): os hedges de cada execucao e um resumo da estabilidade."""
    d = BENCHMARKS[bench]["loader"]().reset_index(drop=True)
    d = d[d.split_orig == "test"].reset_index(drop=True)
    texts, um = d.text.tolist(), d.u_m.tolist()
    prompt = seed_codebook(bench, "blind")
    print(f"{bench}: {len(d)} itens de teste, {k} repeticoes")

    saved = C.OR_PROVIDER["assessor"]
    if provider != "keep": C.OR_PROVIDER["assessor"] = provider
    if C.BACKEND == "openrouter":
        print("provedor fixado:", C.OR_PROVIDER["assessor"])
    runs, provs_all = [], []
    try:
        for i in range(k):
            res = assess_blind_meta(prompt, texts)
            hs, provs = [h for h, _ in res], [p for _, p in res]
            runs.append(hs); provs_all.append(provs)
            print(f"run {i}: rho_u={M.spearman(hs, um):+.3f}  cobertura={M.coverage(hs):.0%}  "
                  f"provedores={dict(collections.Counter(provs))}")
    finally:
        C.OR_PROVIDER["assessor"] = saved

    pd.DataFrame({"item_id": d.item_id, "text": texts,
                  **{f"hedge_{i}": r for i, r in enumerate(runs)},
                  **{f"provider_{i}": p for i, p in enumerate(provs_all)}}
                 ).to_csv(f"{C.RESDIR}/noise_{bench}_blind.csv", index=False)

    A = np.array([[np.nan if h is None else h for h in r] for r in runs])        # K x n
    ok = ~np.isnan(A).any(axis=0)                                               # validos em todas
    rho = np.array([M.spearman(r, um) for r in runs])
    pair = [np.mean(A[i, ok] == A[j, ok]) for i in range(len(A)) for j in range(i + 1, len(A))]
    diff = [abs(rho[i] - rho[j]) for i in range(len(rho)) for j in range(i + 1, len(rho))]
    summary = pd.Series({
        "n_items_validos": int(ok.sum()),
        "itens_que_mudam": float(np.mean([len(set(c)) > 1 for c in A[:, ok].T])),
        "concordancia_par_a_par": float(np.mean(pair)),
        "rho_u_media": rho.mean(), "rho_u_dp": rho.std(ddof=1) if len(rho) > 1 else float("nan"),
        "rho_u_amplitude": rho.max() - rho.min(),
        "max_dif_par_a_par": max(diff) if diff else float("nan")})
    return runs, summary


# ---- piso de ruido do assessor nos prompts que entram nas tabelas ---------------------------
def _floor_path(bench, cond, method):
    return f"{C.RESDIR}/floor_{bench}_{cond}_{method}.csv"


def floor_test(frames, codebooks, bench, conditions=None, methods=("single_hedge", "GEPA"), k=3):
    """Repete `k` vezes, no split de teste, a avaliacao de cada (condicao, metodo): as mesmas chamadas
    ao assessor, temperatura 0, para medir quanto o hedge varia entre execucoes identicas. Os metodos
    cujo prompt ainda nao existe (ex.: GEPA antes do codebook) sao pulados. Um CSV por
    (benchmark, condicao, metodo) com o hedge de cada passada; refazer so recalcula o que falta."""
    f = frames[bench]; test = M.test_fold_with_human(f[f.split_orig == "test"])
    for cond in (conditions or C.CONDITIONS):
        prompts = dict(baselines(bench, cond))
        if f"{bench}__{cond}" in codebooks:
            prompts["GEPA"] = codebooks[f"{bench}__{cond}"]
        for name in methods:
            if name not in prompts:
                print(f"{bench} {cond} {name}: sem prompt ainda, pulando"); continue
            path = _floor_path(bench, cond, name)
            if os.path.exists(path) and sum(c.startswith("hedge_") for c in pd.read_csv(path, nrows=1).columns) >= k:
                print(f"{bench} {cond} {name}: {k} passadas ja em {os.path.basename(path)}, pulando"); continue
            out = {"item_id": test["ids"], "u_m": test["um"]}
            for i in range(k):
                h, _ = run_assessor(prompts[name], test, cond, bench)
                out[f"hedge_{i}"] = h
                print(f"{bench:12} {cond:9} {name:13} passada {i}: rho_u={M.spearman(h, test['um']):+.3f} cov={M.coverage(h):.0%}")
            pd.DataFrame(out).to_csv(path, index=False)
    return floor_summary(bench)


def floor_summary(bench):
    """Resumo de todos os floor_<bench>_*.csv em disco: ρ por passada, itens que mudam e a diferenca
    entre passadas identicas com intervalo bootstrap (4000 reamostragens e semente fixa, como no paper)."""
    rows = []
    for path in sorted(glob.glob(f"{C.RESDIR}/floor_{bench}_*_*.csv")):
        _, _, cond, method = os.path.basename(path)[:-4].split("_", 3)
        d = pd.read_csv(path); cols = [c for c in d.columns if c.startswith("hedge_")]
        H = [[None if pd.isna(x) else int(x) for x in d[c]] for c in cols]
        um = d.u_m.tolist(); rho = np.array([M.spearman(h, um) for h in H])
        A = np.array([[np.nan if x is None else x for x in h] for h in H]); ok = ~np.isnan(A).any(axis=0)
        flips = float(np.mean([len(set(c)) > 1 for c in A[:, ok].T]))
        rng = np.random.default_rng(C.SEED); hh = [np.array(h, dtype=float) for h in H]; uu = np.array(um)
        best = (0.0, (float("nan"), float("nan")))
        for i in range(len(H)):
            for j in range(i + 1, len(H)):
                delta = rho[i] - rho[j]
                ok2 = ~np.isnan(hh[i]) & ~np.isnan(hh[j]); n = int(ok2.sum()); idx = np.arange(len(uu))[ok2]
                B = []
                for _ in range(4000):
                    r = rng.choice(idx, n)
                    B.append(M.spearman(hh[i][r], uu[r]) - M.spearman(hh[j][r], uu[r]))
                if abs(delta) >= abs(best[0]):
                    best = (delta, tuple(np.percentile(B, [2.5, 97.5])))
        rows.append({"benchmark": bench, "condition": cond, "method": method, "k": len(H), "n_items": int(ok.sum()),
                     "rho_mean": rho.mean(), "rho_sd": rho.std(ddof=1) if len(rho) > 1 else float("nan"),
                     "rho_range": rho.max() - rho.min(), "items_flip": flips,
                     "max_abs_diff": abs(best[0]), "diff_ci_lo": best[1][0], "diff_ci_hi": best[1][1]})
    out = pd.DataFrame(rows)
    if len(out): out.to_csv(f"{C.RESDIR}/floor_summary_{bench}.csv", index=False)
    return out
