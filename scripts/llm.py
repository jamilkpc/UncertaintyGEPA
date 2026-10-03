"""Chamadas aos modelos: HTTP com retry, LLM 1 (anotador), LLM 2 (assessor) e LLM 3 (mestre)."""
import collections
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

import config as C
from benchmarks import BENCHMARKS

TRANSIENT = {401, 408, 429, 500, 502, 503, 504}   # 401 do Cloudflare costuma ser carga, nao auth


def _post(url, payload, retries=6, timeout=None):
    backoff = 1.0
    last = None
    for _ in range(retries):
        try:
            r = requests.post(url, headers=C.HEADERS, json=payload, timeout=timeout or C.REQUEST_TIMEOUT)
        except requests.RequestException as e:
            last = f"rede: {e}"; time.sleep(backoff); backoff *= 2; continue
        if r.ok:
            return r.json()
        last = f"HTTP {r.status_code}: {r.text[:200]}"
        if r.status_code in TRANSIENT:
            time.sleep(float(r.headers.get("Retry-After", backoff))); backoff *= 2; continue
        raise RuntimeError(last)                   # 400, 404: erro de verdade, falha rapido
    raise RuntimeError(f"desisti apos {retries} tentativas. Ultima: {last}")


def _last_dash_block(text):
    """Ultimo bloco delimitado por linhas '---' (o formato que o template do mestre pede), ou None."""
    lines = (text or "").splitlines()
    marks = [i for i, l in enumerate(lines) if l.strip() == "---"]
    if len(marks) < 2:
        return None
    block = "\n".join(lines[marks[-2]:marks[-1] + 1]).strip()
    return block if len(block) > 6 else None


def _content(data, from_reasoning=False):
    """So o content. Por padrao NUNCA cai para o raciocinio: raciocinio truncado viraria
       hedge fantasma com cobertura 100% e nada acusaria.
       from_reasoning=True (so o mestre): se content vier vazio com a geracao COMPLETA
       (finish_reason == 'stop'), pega o ultimo bloco '---' do raciocinio. Bug conhecido do vLLM com
       o Qwen 3.5 (content=None, resposta so no campo de raciocinio). Se nao houver bloco, erra."""
    res = data.get("result", data)
    if isinstance(res, dict) and res.get("response") is not None:
        return str(res["response"])
    ch = (res.get("choices") or [{}])[0] or {}
    msg = ch.get("message", {}) or {}
    content = msg.get("content")
    if not content and from_reasoning and ch.get("finish_reason") == "stop":
        block = _last_dash_block(msg.get("reasoning_content") or msg.get("reasoning"))
        if block:
            print("  [aviso] content vazio; usei o ultimo bloco '---' do raciocinio")
            return block
    if not content:
        raise RuntimeError(f"content vazio (finish_reason={ch.get('finish_reason')}); "
                           "provavel truncamento - suba max_completion_tokens")
    return str(content)


# ---- LLM 1: anotador --------------------------------------------------------------------------
def annotate_once(text, temperature, bench):
    b = BENCHMARKS[bench]
    out = _post(C.URLS["annotator"], C.chat_payload(
        "annotator",
        [{"role": "system", "content": b["ann"]},
         {"role": "user",   "content": f"ITEM:\n{text}"}],
        temperature, 4))
    m = re.search(r"\d", _content(out))
    if not m: return None
    v = int(m.group())
    return v if v in b["levels"] else None


def draws_variance(draws):
    """Variancia das R extracoes. Binario -> p(1-p); ordinal -> a mesma expressao."""
    v = [d for d in draws if d is not None]
    return float(np.var(np.asarray(v, dtype=float))) if v else float("nan")


def run_annotator(frame, bench, chunk=100):
    """Grava a cada `chunk` itens e retoma do cache. Uma queda no meio nao custa a passada."""
    cache = f"{C.WORKDIR}/llm1_{bench}.csv"
    done = set(pd.read_csv(cache).item_id) if os.path.exists(cache) else set()
    todo = frame[~frame.item_id.isin(done)]
    if done:
        print(f"  {bench}: {len(done)} ja em cache, faltam {len(todo)}")

    def one(row):
        try:
            lab = annotate_once(row.text, 0.0, bench)
            with ThreadPoolExecutor(max_workers=4) as ex:
                draws = list(ex.map(lambda _: annotate_once(row.text, C.TEMP_RESAMPLE, bench),
                                    range(C.R)))
        except Exception as e:                      # item problematico nao derruba o bloco
            print(f"  [{row.item_id}] {str(e)[:90]}")
            return {"item_id": row.item_id, "llm1_label": None, "g_m": float("nan"),
                    "n_valid_draws": 0}
        return {"item_id": row.item_id, "llm1_label": lab, "g_m": draws_variance(draws),
                "n_valid_draws": sum(d is not None for d in draws)}

    for i in range(0, len(todo), chunk):
        part = todo.iloc[i:i + chunk]
        with ThreadPoolExecutor(max_workers=C.WORKERS) as ex:
            rows = list(ex.map(one, list(part.itertuples())))
        pd.DataFrame(rows).to_csv(cache, mode="a", index=False, header=not os.path.exists(cache))
        print(f"  {bench}: {min(i + chunk, len(todo))}/{len(todo)}")
    return pd.read_csv(cache)


# ---- LLM 2: assessor --------------------------------------------------------------------------
_HEDGE_RE = [(h, re.compile(rf"(?<!\w){re.escape(h)}(?!\w)"))
             for h in sorted(C.HEDGES, key=len, reverse=True)]


def parse_hedge(text):
    lines = [l.strip().lower().strip(".*_`\"' ") for l in (text or "").splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        if last in C.HEDGE_TO_ORD: return C.HEDGE_TO_ORD[last]
        for h, rx in _HEDGE_RE:
            if rx.search(last): return C.HEDGE_TO_ORD[h]
    low = (text or "").lower(); best_pos, best = -1, None
    for h, rx in _HEDGE_RE:
        hits = list(rx.finditer(low))
        if hits and hits[-1].start() > best_pos: best_pos, best = hits[-1].start(), C.HEDGE_TO_ORD[h]
    return best


def user_msg(text, label, condition, bench):
    if condition == "labelled":
        return f"ITEM:\n{text}\n\nASSIGNED LABEL: {BENCHMARKS[bench]['verb'].get(label, label)}"
    return f"ITEM:\n{text}"


def run_assessor(codebook, fold, condition, bench):
    prompt = codebook.strip() + "\n\n" + C.FIXED_OUTPUT_LINE
    def one(t):
        try:
            out = _post(C.URLS["assessor"], C.chat_payload(
                "assessor",
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": user_msg(t[0], t[1], condition, bench)}],
                0.0, 1024))
            raw = _content(out)
            return parse_hedge(raw), raw
        except Exception as e:
            return None, f"__error__: {e}"
    with ThreadPoolExecutor(max_workers=C.WORKERS) as ex:
        res = list(ex.map(one, list(zip(fold["texts"], fold["labels"]))))
    return [h for h, _ in res], [o for _, o in res]


def assess_blind_meta(prompt, texts):
    """Assessor blind, temperatura 0. Devolve (hedge, provedor que serviu) por item."""
    full = prompt.strip() + "\n\n" + C.FIXED_OUTPUT_LINE
    def one(t):
        try:
            out = _post(C.URLS["assessor"], C.chat_payload(
                "assessor",
                [{"role": "system", "content": full}, {"role": "user", "content": f"ITEM:\n{t}"}],
                0.0, 1024))
            return parse_hedge(_content(out)), out.get("provider")
        except Exception:
            return None, "__error__"
    with ThreadPoolExecutor(max_workers=C.WORKERS) as ex:
        return list(ex.map(one, texts))


# ---- LLM 3: mestre ----------------------------------------------------------------------------
CONTEXT = {"key": None}      # preenchido por gepa_search antes de cada otimizacao (ex.: "ConvAbuse__blind")


def _log_master(rec):
    """Uma linha JSON por chamada do mestre em <experimento>/work/master_calls.jsonl."""
    try:
        with open(f"{C.WORKDIR}/master_calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _master_usage(data, t0, prompt, max_tokens, reasoning, fallback, error=None):
    res = data.get("result", data) if isinstance(data, dict) else {}
    ch = ((res.get("choices") or [{}])[0] or {}) if isinstance(res, dict) else {}
    msg = ch.get("message", {}) or {}
    cot = msg.get("reasoning_content") or msg.get("reasoning") or ""
    u = res.get("usage", {}) or {} if isinstance(res, dict) else {}
    return {"ts": time.strftime("%H:%M:%S"), "key": CONTEXT["key"], "seconds": round(time.time() - t0, 1),
            "fallback_sem_raciocinio": fallback, "raciocinio": bool(reasoning), "budget": C.MASTER_THINKING_BUDGET,
            "max_tokens": max_tokens, "prompt_chars": len(prompt), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "reasoning_chars": len(cot), "finish_reason": ch.get("finish_reason"),
            # o servidor injeta esta frase quando o orcamento de raciocinio acaba (--reasoning-config)
            "budget_atingido": ("directly now" in cot) if cot else None, "erro": error}


def _fit_max_tokens(prompt, want):
    """No vLLM prompt + max_tokens tem de caber no contexto do servidor, senao ha HTTP 400."""
    if C.BACKEND != "vllm":
        return want
    est_prompt = int(len(prompt) / 2.8) + 300          # estimativa conservadora de tokens do prompt
    return max(2000, min(want, C.SERVER_CONTEXT - est_prompt))


def master_lm(prompt):
    """Chamada do mestre. Primeiro com raciocinio e o teto de tokens grande; se estourar o teto ou
    falhar por qualquer motivo, repete a MESMA chamada sem raciocinio (curta e confiavel), para
    uma rodada ruim nao derrubar o GEPA (que roda com raise_on_exception=True).
    Cada chamada e registrada em work/master_calls.jsonl (tokens usados, se o orcamento foi atingido)."""
    if isinstance(prompt, list):
        prompt = "\n\n".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in prompt)
    msgs = [{"role": "user", "content": prompt}]
    mt = _fit_max_tokens(prompt, C.MASTER_MAX_TOKENS); t0 = time.time(); data = {}
    try:
        data = _post(C.URLS["master"], C.chat_payload("master", msgs, 1.0, mt, reasoning=C.MASTER_EFFORT),
                     retries=2, timeout=C.MASTER_TIMEOUT)
        out = _content(data, from_reasoning=True).strip()
        _log_master(_master_usage(data, t0, prompt, mt, C.MASTER_EFFORT, False))
        return out
    except RuntimeError as e:
        _log_master(_master_usage(data, t0, prompt, mt, C.MASTER_EFFORT, False, error=str(e)[:160]))
        if C.MASTER_EFFORT is None:
            raise
        print(f"  [aviso] mestre com raciocinio falhou ({str(e)[:90]}); repetindo sem raciocinio")
        t1 = time.time(); mt2 = _fit_max_tokens(prompt, 4000)
        data2 = _post(C.URLS["master"], C.chat_payload("master", msgs, 1.0, mt2, reasoning=None),
                      retries=2, timeout=C.MASTER_TIMEOUT)
        out = _content(data2).strip()
        _log_master(_master_usage(data2, t1, prompt, mt2, None, True))
        return out
