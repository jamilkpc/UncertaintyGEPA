"""Configuracao global: backend de LLM, modelos, constantes e diretorios de saida.

Os outros modulos leem tudo por `import config as C` e `C.NOME`, nunca `from config import NOME`,
para que `configure()` (troca de backend no meio da sessao) valha em todo lugar.
"""
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "paper_materials"
LLM1_DIR = str(OUTDIR / "inputs" / "llm1")   # entrada: llm1_<tag>_<benchmark>.csv ja gerados
WORKDIR = PAPERDIR = RESDIR = None


def set_experiment(name=None, create=True):
    """Isola as saidas de um experimento em paper_materials/<name>/{work,results,paper}, para um
    experimento novo nao reaproveitar codebooks nem sobrescrever o summary de outro.
    Sem nome: paper_materials/{work,results,paper}."""
    global WORKDIR, PAPERDIR, RESDIR
    base = OUTDIR / name if name else OUTDIR
    WORKDIR, PAPERDIR, RESDIR = (str(base / d) for d in ("work", "paper", "results"))
    if create:
        for d in (WORKDIR, PAPERDIR, RESDIR):
            os.makedirs(d, exist_ok=True)
    return str(base)


set_experiment(create=False)      # so define os caminhos; as pastas sao criadas quando um experimento e escolhido

try:      # macOS: Python sem CAs do sistema quebra o download dos corpora (CERTIFICATE_VERIFY_FAILED)
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

try:
    from google.colab import userdata
    def secret(k): return userdata.get(k)
except Exception:
    def secret(k): return os.environ[k]

# ---- modelos por backend --------------------------------------------------------------------
OR_MODELS = {
    "annotator": os.environ.get("OR_ANNOTATOR", "google/gemma-4-26b-a4b-it"),
    "assessor":  os.environ.get("OR_ASSESSOR",  "z-ai/glm-4.7-flash"),
    "master":    os.environ.get("OR_MASTER",    "z-ai/glm-5.2"),
}
OR_PROVIDER = {"annotator": None, "assessor": "cloudflare", "master": "siliconflow/fp8"}  # fixar
OR_SEED = 42

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
OLLAMA_MODELS = {r: os.environ.get(f"OLLAMA_{r.upper()}", OLLAMA_MODEL)
                 for r in ("annotator", "assessor", "master")}
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")

# vLLM (servidor proprio, ex.: a 4090 do laboratorio, acessado por tunel SSH em localhost:8000)
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")

REQUEST_TIMEOUT = 180      # segundos por chamada (assessor, anotador)
MASTER_TIMEOUT  = int(os.environ.get("MASTER_TIMEOUT", 600))      # teto (s) por chamada do mestre; nao e o tempo esperado
MASTER_MAX_TOKENS = int(os.environ.get("MASTER_MAX_TOKENS", 8000))  # com raciocinio ligado o Qwen pode gastar mais de 8000 so pensando
MASTER_THINKING_BUDGET = int(os.environ.get("MASTER_THINKING_BUDGET", 0)) or None   # vLLM: tokens de raciocinio do mestre (exige --reasoning-config no servidor)
HUMAN_TAG = "human"        # --llm1-tag human: rotulo = consenso humano, alvo = desacordo humano u_m (variante supervisionada)
TARGET_KIND = "llm1"       # "llm1" ou "human"; definido por run_pipeline a partir de --llm1-tag
TARGET_SOURCE = "resampling"   # alvo g_m da LLM 1: "resampling" (variancia das R amostras) ou "logprob" (variancia sob as probabilidades dos rotulos)
SERVER_CONTEXT = int(os.environ.get("SERVER_CONTEXT", 16384))   # --max-model-len do servidor vLLM; limita o max_tokens do mestre
MASTER_EFFORT = "high"     # esforco de raciocinio do mestre (a LLM 2 nunca raciocina); $MASTER_EFFORT sobrescreve

# ---- escala de hedges (a ultima linha do prompt do assessor e fixa) ----------------------------
HEDGES = ["not uncertain", "somewhat uncertain", "highly uncertain"]
H = len(HEDGES)
HEDGE_TO_ORD = {h: i for i, h in enumerate(HEDGES)}
FIXED_OUTPUT_LINE = ("On the LAST line, write EXACTLY one of the following and nothing else: "
                     + " | ".join(HEDGES) + ".")

# ---- desenho experimental ---------------------------------------------------------------------
SEED             = 42
R                = 20
TEMP_RESAMPLE    = 1.0
DEFAULT_WORKERS  = 12
N_TRAIN_FOLDS    = 6
N_VAL_FOLDS      = 2
N_BOOT           = 2000
MAX_METRIC_CALLS = 60
COVERAGE_FLOOR   = 0.90
CONDITIONS       = ["labelled", "blind"]

# ---- preenchidos por configure() ----------------------------------------------------------------
BACKEND = None
HEADERS = None
URLS = None
WORKERS = DEFAULT_WORKERS


def derive_experiment_name(llm1_tag, backend, target="resampling"):
    """Nome padrao: <llm1>__<modelo>__<raciocinio do mestre>, ex.: gpt56luna__qwen3.5-9b__think8k.
    O nome descreve a CONFIGURACAO, nao o benchmark: todos os benchmarks rodados com a mesma configuracao
    ficam na mesma pasta (e a trava de configuracao impede misturar configuracoes diferentes)."""
    model = {"vllm": VLLM_MODEL, "ollama": OLLAMA_MODELS["assessor"], "openrouter": OR_MODELS["assessor"]}.get(backend, backend)
    model = model.split("/")[-1].lower().replace(":", "-")
    eff = os.environ.get("MASTER_EFFORT")
    if eff == "none":
        think = "think-off"
    elif MASTER_THINKING_BUDGET:
        think = f"think{MASTER_THINKING_BUDGET // 1000}k" if MASTER_THINKING_BUDGET % 1000 == 0 else f"think{MASTER_THINKING_BUDGET}"
    else:
        think = "think-free"
    src = f"{llm1_tag or 'llm1-gerada'}" + ("-logprob" if target == "logprob" else "")
    suffix = os.environ.get("EXP_SUFFIX")      # ex.: hedgefix -> pasta nova, sem reaproveitar cache de runs anteriores
    return f"{src}__{model}__{think}" + (f"__{suffix}" if suffix else "")


def configure(backend=None):
    """Escolhe o backend: "cloudflare", "openrouter", "ollama" ou "vllm". Sem argumento, le $BACKEND
    (padrao "ollama"). Pode ser chamada de novo para trocar de backend na mesma sessao."""
    global BACKEND, HEADERS, URLS, WORKERS, MASTER_EFFORT
    backend = backend or os.environ.get("BACKEND", "ollama")
    if backend == "cloudflare":
        gw = (f"https://gateway.ai.cloudflare.com/v1/{secret('CF_Palver_1')}/default/workers-ai")
        HEADERS = {"Content-Type": "application/json", "Authorization": secret("CF_Palver_2")}
        # sem cf-aig-skip-cache: prompt identico devolve resposta cacheada, o que torna
        # reexecucoes reprodutiveis apesar do nao-determinismo de serving em temperatura 0.
        URLS = {"annotator": f"{gw}/@cf/google/gemma-4-26b-a4b-it",
                "assessor":  f"{gw}/@cf/zai-org/glm-4.7-flash",
                "master":    f"{gw}/@cf/zai-org/glm-5.2"}
    elif backend == "openrouter":
        HEADERS = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {secret('OPEN_ROUTER_API_KEY')}"}
        URLS = {k: "https://openrouter.ai/api/v1/chat/completions" for k in OR_MODELS}
    elif backend == "ollama":
        HEADERS = {"Content-Type": "application/json"}
        URLS = {k: OLLAMA_URL for k in OLLAMA_MODELS}
        try:
            requests.get(OLLAMA_URL.split("/v1/")[0] + "/api/tags", timeout=3).raise_for_status()
        except Exception as e:
            raise RuntimeError(f"BACKEND='ollama', mas o servidor nao responde em {OLLAMA_URL} ({e}). "
                               "Suba o Ollama (`ollama serve`) ou use 'cloudflare'/'openrouter'.")
    elif backend == "vllm":
        HEADERS = {"Content-Type": "application/json"}
        URLS = {k: VLLM_URL for k in ("annotator", "assessor", "master")}
        try:
            r = requests.get(VLLM_URL.split("/chat/")[0] + "/models", timeout=5); r.raise_for_status()
            served = [m["id"] for m in r.json().get("data", [])]
        except Exception as e:
            raise RuntimeError(f"BACKEND='vllm', mas o servidor nao responde em {VLLM_URL} ({e}). "
                               "Suba o `vllm serve` e abra o tunel SSH (ver scripts/README).")
        if VLLM_MODEL not in served:
            raise RuntimeError(f"vLLM serve {served}, mas VLLM_MODEL={VLLM_MODEL!r}. Defina $VLLM_MODEL.")
    else:
        raise ValueError(f"BACKEND desconhecido: {backend}")
    BACKEND = backend
    # $MASTER_EFFORT sobrescreve: "none" desliga o raciocinio do mestre, ou "low"/"medium"/"high".
    env_effort = os.environ.get("MASTER_EFFORT")
    MASTER_EFFORT = (None if env_effort == "none" else env_effort) if env_effort else "high"
    WORKERS = {"ollama": lambda: int(os.environ.get("OLLAMA_WORKERS", 1)),
               "vllm": lambda: int(os.environ.get("VLLM_WORKERS", 16))}.get(backend, lambda: DEFAULT_WORKERS)()
    return BACKEND


def chat_payload(role, messages, temperature, max_tokens, reasoning=None):
    """Monta o corpo da requisicao no dialeto do backend. reasoning: None (desligado) ou esforco."""
    if BACKEND == "cloudflare":
        p = {"messages": messages, "temperature": temperature, "max_completion_tokens": max_tokens}
        if reasoning: p["reasoning_effort"] = reasoning
        else:         p["chat_template_kwargs"] = {"enable_thinking": False}
        return p
    if BACKEND == "vllm":
        # raciocinio por requisicao via chat_template_kwargs (o assessor nunca raciocina; o mestre sim)
        p = {"model": VLLM_MODEL, "messages": messages, "temperature": temperature,
             "max_tokens": max_tokens, "seed": OR_SEED,
             "chat_template_kwargs": {"enable_thinking": bool(reasoning)}}
        if reasoning and role == "master" and MASTER_THINKING_BUDGET:
            p["thinking_token_budget"] = MASTER_THINKING_BUDGET     # campo de topo, nao extra_body
        return p
    if BACKEND == "ollama":
        # reasoning_effort="none" desliga o raciocinio (Qwen 3.5 pensa por padrao e gasta o max_tokens
        # inteiro antes de escrever o content). Testado no qwen3.5:9b; chat_template_kwargs e ignorado.
        return {"model": OLLAMA_MODELS[role], "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "seed": OR_SEED, "reasoning_effort": reasoning or "none"}
    p = {"model": OR_MODELS[role], "messages": messages, "temperature": temperature,
         "max_tokens": max_tokens, "seed": OR_SEED,
         "reasoning": {"effort": reasoning} if reasoning else {"enabled": False}}
    prov = {"require_parameters": True}       # so provedores que honram temperature/seed
    if OR_PROVIDER[role]:
        prov.update(order=[OR_PROVIDER[role]], allow_fallbacks=False)
    p["provider"] = prov
    return p
