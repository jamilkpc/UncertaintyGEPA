"""Metricas, folds e o teto de uma escala de H niveis (Apendice A do paper)."""
import collections
import math

import numpy as np
from scipy.stats import spearmanr

import config as C


def spearman(h, t):
    p = [(a, b) for a, b in zip(h, t) if a is not None]
    if len(p) < 2: return 0.0
    r = spearmanr(*zip(*p)).correlation
    return 0.0 if (r is None or math.isnan(r)) else float(r)


def spearman_ci(h, t, n_boot=None, seed=None):
    n_boot = n_boot or C.N_BOOT
    seed = C.SEED if seed is None else seed
    p = [(a, b) for a, b in zip(h, t) if a is not None]
    if len(p) < 3: return (float("nan"), float("nan"))
    hs, ts = map(np.asarray, zip(*p)); rng = np.random.default_rng(seed); B = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(hs), len(hs))
        if len(set(hs[ix])) < 2 or len(set(ts[ix])) < 2: continue
        B.append(spearmanr(hs[ix], ts[ix]).correlation)
    B = [b for b in B if not math.isnan(b)]
    return tuple(np.percentile(B, [2.5, 97.5])) if B else (float("nan"), float("nan"))


def coverage(h): return sum(x is not None for x in h) / max(1, len(h))


def rho_max_discrete(values, H=None):
    """Teto exato da eq. (1) do Apendice A.3, por programacao dinamica O(K^2 H)."""
    H = H or C.H
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    cnt = collections.Counter(np.round(v, 9)); vals = sorted(cnt); n = len(v)
    pi = [cnt[x] / n for x in vals]; K = len(pi)
    den = 1 - sum(p ** 3 for p in pi)
    if den <= 0: return K, float("nan")
    P = np.concatenate([[0.0], np.cumsum(pi)]); INF = float("inf")
    dp = [[INF] * (K + 1) for _ in range(H + 1)]; dp[0][0] = 0.0
    for j in range(1, H + 1):
        for i in range(1, K + 1):
            dp[j][i] = min((dp[j-1][t] + (P[i]-P[t])**3 for t in range(j-1, i) if dp[j-1][t] < INF),
                           default=INF)
    return K, math.sqrt((1 - min(dp[j][K] for j in range(1, H + 1))) / den)


def train_fold(sub):
    """Fold para o GEPA: nada de origem humana, so o que o LLM 1 produziu."""
    return {"texts": sub.text.tolist(), "labels": sub.llm1_label.tolist(),
            "gm": sub.g_m.tolist(), "ids": sub.item_id.tolist()}


def test_fold_with_human(sub):
    """Fold da avaliacao final: acrescenta o desacordo humano e o rotulo de consenso."""
    return train_fold(sub) | {"um": sub.u_m.tolist(), "consensus": sub.consensus.tolist()}


def stratified_folds(frame, n, seed):
    rng = np.random.default_rng(seed); folds = [[] for _ in range(n)]
    for _, grp in frame.groupby(frame.u_m > 0):
        for j, i in enumerate(rng.permutation(grp.index.to_numpy())): folds[j % n].append(i)
    return [frame.loc[f] for f in folds if f]
