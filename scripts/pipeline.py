"""Dados e LLM 1: carrega os corpora, estima o custo e monta os frames (texto, rotulo, g_m, u_m)."""
import os

import pandas as pd

import config as C
import metrics as M
from benchmarks import BENCHMARKS, RUN
from llm import run_annotator
from prompts import baselines


def load_corpora(run=None):
    """Corpus completo de cada benchmark. So rede para baixar os dados, nenhuma chamada a LLM."""
    return {b: BENCHMARKS[b]["loader"]().reset_index(drop=True) for b in (run or RUN)}


def estimate_cost(data, llm1_given=False):
    """Custo estimado em chamadas. llm1_given=True: a LLM 1 ja vem de arquivo e nao entra na conta."""
    est = []
    for b, d in data.items():
        n = len(d); ntr = (d.split_orig == "train").sum(); nte = (d.split_orig == "test").sum()
        n_meth = len(baselines(b, "labelled")) + len(baselines(b, "blind")) + 2   # +GEPA por condicao
        n_annot = 0 if llm1_given else n * (C.R + 1)
        n_gepa = C.MAX_METRIC_CALLS * (ntr / C.N_TRAIN_FOLDS) * len(C.CONDITIONS)
        n_eval = nte * n_meth
        est.append({"benchmark": b, "itens": n, "u_m>0": f"{(d.u_m > 0).mean():.0%}",
                    "chamadas LLM1": int(n_annot), "chamadas GEPA": int(n_gepa),
                    "chamadas aval.": int(n_eval), "total": int(n_annot + n_gepa + n_eval)})
    cost = pd.DataFrame(est)
    print(cost.to_string(index=False))
    print(f"\nTOTAL GERAL: {cost.total.sum():,} chamadas "
          f"({cost['chamadas LLM1'].sum():,} do LLM 1, pagas uma vez pelo cache em disco)"
          if not llm1_given else f"(LLM 1 ja gerada, nao entra na conta)")
    return cost


def load_llm1_results(bench, tag):
    """Le um llm1_<tag>_<bench>.csv ja gerado (rotulo + R reamostragens). Devolve as mesmas colunas
    de run_annotator: item_id, llm1_label, g_m (variancia das R extracoes), n_valid_draws.
    So `text`, `u_m` e `consensus` vem do loader; as colunas humanas do arquivo sao ignoradas."""
    path = f"{C.LLM1_DIR}/llm1_{tag}_{bench}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    x = pd.read_csv(path)
    n_bad = int((x.status != "complete").sum())
    if n_bad: print(f"  {bench}: {n_bad} itens com status != complete, descartados")
    x = x[x.status == "complete"]
    if C.TARGET_SOURCE == "logprob":
        # alvo = variancia do rotulo sob as probabilidades por opcao (1 chamada em vez de R+1); o rotulo e o mesmo
        print(f"  {bench}: alvo = logprob_var (nao a variancia das amostras)")
        x = x.assign(g_m=x.logprob_var, n_valid_draws=1)
        return x[["item_id", "llm1_label", "g_m", "n_valid_draws"]]
    return (x[["item_id", "llm1_label", "resampling_var", "n_valid_resamples"]]
            .rename(columns={"resampling_var": "g_m", "n_valid_resamples": "n_valid_draws"}))


def human_annotations(d):
    """Versao 'humana' da LLM 1: o rotulo e o consenso dos anotadores e o alvo de instabilidade `g_m` e o desacordo
    humano `u_m` do proprio item. Nenhuma LLM e chamada. ATENCAO: aqui o alvo de treino e a quantidade usada na
    avaliacao, entao rho(h, g) == rho(h, u) e a regra 'nada humano entra na busca' deixa de valer."""
    return pd.DataFrame({"item_id": d.item_id, "llm1_label": d.consensus.astype(int),
                         "g_m": d.u_m.astype(float), "n_valid_draws": d.n_ann})


def build_frames(data, llm1_tag=None):
    """Junta o LLM 1 ao corpus: {benchmark: DataFrame}.
    llm1_tag=None roda o anotador configurado (com cache em disco); com um tag, le os resultados
    ja gerados em `llm1_<tag>_<benchmark>.csv` e nao chama nenhuma LLM."""
    frames = {}
    for b, d in data.items():
        if llm1_tag == C.HUMAN_TAG:
            print(f"{b}: anotadores HUMANOS no lugar da LLM 1 (rotulo = consenso, alvo = desacordo u_m; {len(d)} itens)")
            llm1 = human_annotations(d)
        elif llm1_tag:
            print(f"{b}: LLM 1 lido de llm1_{llm1_tag}_{b}.csv ({len(d)} itens no corpus)")
            llm1 = load_llm1_results(b, llm1_tag)
        else:
            print(f"{b}: LLM 1 em {len(d)} itens")
            llm1 = run_annotator(d, b)
        f = d.merge(llm1, on="item_id").dropna(subset=["llm1_label", "g_m"]).reset_index(drop=True)
        f["llm1_label"] = f.llm1_label.astype(int)
        frames[b] = f
        acc = (f.llm1_label == f.consensus).mean()
        base = f.consensus.value_counts(normalize=True).max()
        print(f"  n={len(f)} acc={acc:.3f} (baseline {base:.3f}) "
              f"g_m=0 em {(f.g_m == 0).mean():.1%} | rho(g,u)={M.spearman(f.g_m, f.u_m):+.3f}")
    return frames
