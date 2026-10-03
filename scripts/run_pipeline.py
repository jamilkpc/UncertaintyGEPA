#!/usr/bin/env python3
"""Roda o pipeline de ponta a ponta, sem notebook: GEPA, avaliacao no teste e .tex do paper.

Feito para rodar direto numa maquina com GPU, dentro de tmux. Tudo e retomavel: codebooks do GEPA
e o summary de cada benchmark ficam em disco, e uma segunda execucao pula o que ja existe.

Exemplo (llm1 ja gerado, um benchmark, vLLM em localhost:8000):
    python scripts/run_pipeline.py --experiment gpt56luna --llm1-tag gpt56luna \
        --benchmarks ConvAbuse --backend vllm

Saidas em paper_materials/<experiment>/{work,results,paper,logs}. Os parametros do desenho
(folds, orcamento do GEPA, R) vivem em config.py e nao tem flag aqui de proposito.
"""
import argparse
import datetime as dt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C                      # noqa: E402
from benchmarks import BENCHMARKS       # noqa: E402

STAGES = ["gepa", "eval", "paper", "floor", "noise"]


class Tee:
    """Duplica stdout/stderr num arquivo de log (com flush a cada escrita)."""
    def __init__(self, stream, path):
        self.stream, self.f = stream, open(path, "a", buffering=1, encoding="utf-8")
    def write(self, s):
        self.stream.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.stream.flush(); self.f.flush()
    def isatty(self):
        return False
    def __getattr__(self, k):
        return getattr(self.stream, k)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, help="nome da pasta de saida em paper_materials/")
    ap.add_argument("--benchmarks", nargs="+", required=True, help=f"de: {', '.join(BENCHMARKS)}")
    ap.add_argument("--llm1-tag", default=None,
                    help="le paper_materials/inputs/llm1/llm1_<tag>_<benchmark>.csv; sem ele, roda o anotador")
    ap.add_argument("--backend", default=os.environ.get("BACKEND", "vllm"),
                    choices=["vllm", "ollama", "openrouter", "cloudflare"])
    ap.add_argument("--conditions", nargs="+", choices=["labelled", "blind"], default=None,
                    help="padrao: as duas. 'blind' e a condicao das tabelas principais")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=["gepa", "eval", "paper"])
    ap.add_argument("--floor-k", type=int, default=3, help="passadas do estagio floor (piso de ruido)")
    ap.add_argument("--floor-methods", nargs="+", default=["single_hedge", "GEPA"],
                    help="metodos do estagio floor; os sem prompt ainda (ex.: GEPA sem codebook) sao pulados")
    ap.add_argument("--noise-bench", default=None, help="benchmark do teste de ruido (padrao: o primeiro)")
    ap.add_argument("--noise-k", type=int, default=3, help="repeticoes do teste de ruido")
    ap.add_argument("--workers", type=int, default=None, help="chamadas simultaneas (sobrescreve o padrao do backend)")
    ap.add_argument("--force-eval", action="store_true", help="reavalia mesmo com summary_<bench>.csv existente")
    ap.add_argument("--no-preflight", action="store_true", help="pula o teste rapido de assessor e mestre")
    ap.add_argument("--dry-run", action="store_true", help="so carrega os dados e mostra o custo; nao chama LLM")
    return ap.parse_args(argv)


def _design_settings(args):
    """Tudo que, se mudar, torna dois runs incomparaveis. Fica em <experimento>/experiment_config.json."""
    return {"backend": args.backend,
            "modelos": {"vllm": C.VLLM_MODEL, "ollama": C.OLLAMA_MODELS, "openrouter": C.OR_MODELS}.get(args.backend),
            "master_effort": C.MASTER_EFFORT, "master_thinking_budget": C.MASTER_THINKING_BUDGET,
            "master_max_tokens": C.MASTER_MAX_TOKENS, "server_context": C.SERVER_CONTEXT,
            "seed": C.SEED, "R": C.R, "n_train_folds": C.N_TRAIN_FOLDS, "n_val_folds": C.N_VAL_FOLDS,
            "max_metric_calls": C.MAX_METRIC_CALLS, "coverage_floor": C.COVERAGE_FLOOR, "llm1_tag": args.llm1_tag}


def check_experiment_config(base, args):
    """Recusa misturar configuracoes no mesmo experimento (ex.: budget 4000 e depois 8000)."""
    import json
    path = f"{base}/experiment_config.json"; now = _design_settings(args)
    if os.path.exists(path):
        old = json.load(open(path))
        diff = {k: (old.get(k), now[k]) for k in now if old.get(k) != now[k]}
        if diff:
            raise SystemExit("este experimento foi criado com outra configuracao (antes -> agora):\n  "
                             + "\n  ".join(f"{k}: {a} -> {b}" for k, (a, b) in diff.items())
                             + "\nUse outro --experiment para nao misturar resultados.")
    else:
        json.dump(now, open(path, "w"), indent=1, ensure_ascii=False)


def stage(name):
    print(f"\n{'=' * 70}\n[{dt.datetime.now():%H:%M:%S}] {name}\n{'=' * 70}", flush=True)


def preflight():
    """Uma chamada ao assessor e uma ao mestre, para falhar cedo (chave, modelo, timeout, content vazio)."""
    import gepa_search, llm
    t = time.time()
    h, raw = llm.run_assessor("An English tweet was labelled for OFFENSIVE LANGUAGE. Judge how uncertain "
                              "a coder deciding this item's label would be.",
                              {"texts": ["lovely weather today"], "labels": [0]}, "blind", "HSBrexitOff")
    if h[0] is None:
        raise SystemExit(f"preflight: assessor nao devolveu hedge valido. Resposta: {raw[0][:300]!r}")
    print(f"  assessor ok ({time.time() - t:.1f}s) hedge={h[0]}")
    t = time.time()
    out = gepa_search.smoke_test_master()
    print(f"  mestre ok ({time.time() - t:.1f}s): {out[:100]!r}")


def main(argv=None):
    args = parse_args(argv)
    unknown = [b for b in args.benchmarks if b not in BENCHMARKS]
    if unknown:
        raise SystemExit(f"benchmarks desconhecidos: {unknown}. Disponiveis: {list(BENCHMARKS)}")

    base = C.set_experiment(args.experiment)
    os.makedirs(f"{base}/logs", exist_ok=True)
    log = f"{base}/logs/run_{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    print(f"log: {log}\nargs: {vars(args)}")

    if args.conditions:
        C.CONDITIONS = list(args.conditions)
    run = list(args.benchmarks)
    t0 = time.time()

    import evaluation, gepa_search, paper, pipeline

    if args.dry_run:
        stage("dry-run: dados e custo (nenhuma chamada a LLM)")
        pipeline.estimate_cost(pipeline.load_corpora(run), llm1_given=bool(args.llm1_tag))
        return

    C.configure(args.backend)
    if args.workers:
        C.WORKERS = args.workers
    print(f"backend={C.BACKEND} workers={C.WORKERS} conditions={C.CONDITIONS} stages={args.stages}")
    if not args.no_preflight and any(s in args.stages for s in ("gepa", "eval")):
        stage("preflight")
        preflight()

    frames = codebooks = None
    if any(s in args.stages for s in ("gepa", "eval", "paper", "floor")):
        stage("dados e LLM 1")
        data = pipeline.load_corpora(run)
        pipeline.estimate_cost(data, llm1_given=bool(args.llm1_tag))
        frames = pipeline.build_frames(data, llm1_tag=args.llm1_tag)

    if "gepa" in args.stages:
        check_experiment_config(base, args)
        stage("GEPA: um codebook por benchmark e condicao")
        codebooks = gepa_search.optimize_codebooks(frames, run, C.CONDITIONS)

    if "eval" in args.stages:
        stage("avaliacao no teste")
        codebooks = codebooks or paper.load_codebooks()
        missing = [f"{b}__{c}" for b in run for c in C.CONDITIONS if f"{b}__{c}" not in codebooks]
        if missing:
            raise SystemExit(f"faltam codebooks para a avaliacao: {missing}. Rode o estagio 'gepa'.")
        for b in run:
            if os.path.exists(f"{C.RESDIR}/summary_{b}.csv") and not args.force_eval:
                print(f"{b}: summary_{b}.csv ja existe, pulando (use --force-eval para refazer)")
                continue
            evaluation.evaluate_test(frames, codebooks, [b])

    if "floor" in args.stages:
        stage("piso de ruido: mesma avaliacao repetida no teste")
        codebooks = codebooks or paper.load_codebooks()
        for b in run:
            print(evaluation.floor_test(frames, codebooks, b, C.CONDITIONS, args.floor_methods, args.floor_k).to_string())

    if "paper" in args.stages:
        stage("tabelas .tex")
        paper.write_all(frames, run=run)

    if "noise" in args.stages:
        stage("teste de ruido do assessor")
        _, summary = evaluation.noise_test(args.noise_bench or run[0], k=args.noise_k)
        print(summary.to_string())

    print(f"\nconcluido em {(time.time() - t0) / 60:.1f} min. Saidas em {base}")


if __name__ == "__main__":
    main()
