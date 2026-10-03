"""GEPA: o mestre (LLM 3) reescreve o codebook do assessor maximizando rho(hedge, g_m).

Invariante: nada de origem humana (u_m, consensus) entra no objetivo, no feedback reflexivo
nem no prompt do mestre. So texto, rotulo do LLM 1 e g_m.
"""
import json
import math
import os

import config as C
import metrics as M
import llm
from llm import master_lm, run_assessor
from prompts import REFLECTION_TEMPLATE, seed_codebook


class AssessorAdapter:
    propose_new_texts = None

    def __init__(self, condition, bench):
        self.condition, self.bench = condition, bench

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch
        outs, scores, trajs = [], [], []
        for fold in batch:
            hedges, _ = run_assessor(candidate["codebook"], fold, self.condition, self.bench)
            cov = M.coverage(hedges)
            rho = M.spearman(hedges, fold["gm"])          # ALVO DE TREINO = g_m
            outs.append({"hedges": hedges})
            scores.append(rho if cov >= C.COVERAGE_FLOOR else 0.0)
            if capture_traces:
                gmax = max([g for g in fold["gm"] if not math.isnan(g)] or [1e-9]) or 1e-9
                worst = []
                for t, lab, h, g in zip(fold["texts"], fold["labels"], hedges, fold["gm"]):
                    d = 9.9 if h is None else abs(h / (C.H - 1) - g / gmax)
                    worst.append((d, t, lab, h, g))
                worst.sort(key=lambda r: -r[0])
                trajs.append({"rho": rho, "cov": cov, "worst": worst[:8]})
        return EvaluationBatch(outputs=outs, scores=scores,
                               trajectories=trajs if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        data = {}
        for comp in components_to_update:
            rows = []
            for tr in eval_batch.trajectories:
                head = (f"Rank correlation on this fold: {tr['rho']:.3f} (coverage {tr['cov']:.0%}).\n"
                        "The items below are where the assessor diverged most from how unstable the "
                        "annotator actually was across repeated draws.\n")
                for d, t, lab, h, g in tr["worst"]:
                    said = C.HEDGES[h] if isinstance(h, int) else "NO VALID HEDGE"
                    truth = ("the annotator varied across draws" if g > 0
                             else "the annotator returned the same label on every draw")
                    rows.append({"Inputs": f"ITEM: {t[:400]}\nASSIGNED LABEL: {lab}",
                                 "Generated Outputs": said,
                                 "Feedback": head + f"On this item {truth}, but the assessor said '{said}'."})
            data[comp] = rows
        return data


def smoke_test_master():
    """Uma chamada curta ao mestre, para pegar erro de chave/modelo antes de gastar o GEPA."""
    return master_lm("Reply with one short sentence: this is a smoke test.")


def optimize_codebooks(frames, run=None, conditions=None):
    """Um codebook por benchmark x condicao. Reaproveita `work/codebook_<bench>__<cond>.txt`."""
    import gepa
    assert hasattr(gepa, "optimize"), f"'gepa' virou {type(gepa)}; reimporte o modulo"
    run = run or list(frames)
    conditions = conditions or C.CONDITIONS
    codebooks = {}
    for b in run:
        f = frames[b]
        tr = [M.train_fold(x) for x in M.stratified_folds(f[f.split_orig == "train"], C.N_TRAIN_FOLDS, C.SEED)]
        va = [M.train_fold(x) for x in M.stratified_folds(f[f.split_orig == "dev"], C.N_VAL_FOLDS, C.SEED + 1)]
        for cond in conditions:
            key = f"{b}__{cond}"; path = f"{C.WORKDIR}/codebook_{key}.txt"
            if os.path.exists(path):
                codebooks[key] = open(path, encoding="utf-8").read(); print(f"{key}: cache"); continue
            print(f"\n=== {key} ===")
            llm.CONTEXT["key"] = key
            res = gepa.optimize(seed_candidate={"codebook": seed_codebook(b, cond)},
                                trainset=tr, valset=va, adapter=AssessorAdapter(cond, b),
                                reflection_lm=master_lm, candidate_selection_strategy="pareto",
                                reflection_prompt_template=REFLECTION_TEMPLATE,
                                reflection_minibatch_size=1, max_metric_calls=C.MAX_METRIC_CALLS,
                                display_progress_bar=True, raise_on_exception=True, seed=C.SEED,
                                run_dir=f"{C.WORKDIR}/gepa_runs/{key}")   # estado do GEPA: permite retomar
            codebooks[key] = res.best_candidate["codebook"]
            open(path, "w", encoding="utf-8").write(codebooks[key])
            try:      # trajetoria completa: todos os candidatos e seus scores de validacao
                json.dump(res.to_dict(), open(f"{C.WORKDIR}/trajectory_{key}.json", "w"), default=str)
            except Exception as e:
                print(f"  (trajetoria nao salva: {e})")
    return codebooks
