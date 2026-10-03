# scripts/

Pipeline em módulos, mais um notebook (`run.ipynb`) para olhar resultados e um runner de linha de comando
(`run_pipeline.py`) para execuções longas.

## Rodar numa máquina com GPU

### 1. Servidor vLLM (fp8 dinâmico, em `tmux`)
```bash
python3 -m venv ~/venv-vllm && source ~/venv-vllm/bin/activate
pip install -U vllm
tmux new -s vllm
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen3.5-9B --host 127.0.0.1 --port 8000 \
  --quantization fp8 --max-model-len 16384 --language-model-only \
  --reasoning-parser qwen3 --gpu-memory-utilization 0.65 --max-num-seqs 32 --seed 42
```
- `--host 127.0.0.1`: o vLLM escuta em todas as interfaces por padrão. Anote `vllm --version`.
- `VLLM_USE_FLASHINFER_SAMPLER=0`: evita a compilação do sampler do FlashInfer, que pode falhar quando o
  `nvcc` do sistema é antigo. Se o seu CUDA toolkit for recente, pode omitir.
- `--gpu-memory-utilization`: fração da memória da GPU reservada ao servidor. Reduza se a GPU for
  compartilhada; com a GPU livre, pode subir.

### 2. Projeto (outro venv, ou o mesmo)
```bash
# da sua máquina local, copiando também os dados da LLM 1 (não estão no git):
rsync -av --exclude .git --exclude paper_materials --exclude __pycache__ --exclude '*.ipynb' \
  ./ user@gpu-host:~/UncertaintyGEPA/
rsync -av paper_materials/inputs/ user@gpu-host:~/UncertaintyGEPA/paper_materials/inputs/

# no servidor:
cd ~/UncertaintyGEPA && python3 -m venv ~/venv-gepa && source ~/venv-gepa/bin/activate
pip install -r scripts/requirements.txt
```
O `rsync` tem de rodar na máquina de origem, não no servidor.

### 3. Rodar (em outro `tmux`)
```bash
tmux new -s gepa
python scripts/run_pipeline.py --dry-run --experiment myrun --llm1-tag mytag --benchmarks ConvAbuse
python -u scripts/run_pipeline.py --experiment myrun --llm1-tag mytag \
    --benchmarks ConvAbuse --backend vllm
```
Sair do tmux sem matar: `Ctrl-b d`. Voltar: `tmux attach -t gepa`.

- Retomável: codebooks (`work/codebook_*.txt`, estado em `work/gepa_runs/`) e `results/summary_<bench>.csv`
  ficam em disco, e uma nova execução pula o que existe (`--force-eval` refaz a avaliação).
- Estágios: `--stages gepa eval paper floor noise`. Só uma condição: `--conditions blind`.
- Log: `paper_materials/<experiment>/logs/run_*.log`.

### Ajustes do mestre e experimentos separados
O mestre (LLM 3) raciocina por padrão. Variáveis de ambiente (valem no vLLM; o servidor precisa de `--reasoning-config`):

| Variável | Efeito |
|---|---|
| `MASTER_THINKING_BUDGET` | tokens de raciocínio por chamada (ex.: 4000, 8000) |
| `MASTER_MAX_TOKENS` | teto total da resposta; reduzido sozinho para caber no contexto do servidor (`SERVER_CONTEXT`, padrão 16384) |
| `MASTER_EFFORT=none` | desliga o raciocínio do mestre |

Cada chamada do mestre é registrada em `<experimento>/work/master_calls.jsonl` (tokens, tempo, se o orçamento
foi atingido, se caiu para o modo sem raciocínio). Cada experimento guarda suas configurações em
`experiment_config.json` e **recusa rodar** com configurações diferentes: para testar outro orçamento, use outro
`--experiment`.

### 3b. Piso de ruído do assessor
Mede quanto o hedge varia entre execuções idênticas (mesmo prompt, mesmo item, temperatura 0). Não precisa
de GEPA nem de codebooks: usa o prompt semente. Pode rodar em outra janela do tmux enquanto o run principal
anda (~11 min; `--workers 8` para disputar menos o servidor):
```bash
python -u scripts/run_pipeline.py --experiment myrun --llm1-tag mytag \
    --benchmarks ConvAbuse --conditions blind --stages floor \
    --floor-methods single_hedge --floor-k 3 --workers 8
```
Saídas: `results/floor_<bench>_<cond>_<método>.csv` (hedge de cada passada) e
`results/floor_summary_<bench>.csv` (itens que mudam, maior diferença de ρ entre passadas e o intervalo
bootstrap dela). O `.tex` (`tab_floor.tex`) sai com `--stages paper`, que **precisa do summary da
avaliação**, então rode-o depois do run principal. Para o codebook otimizado: `--floor-methods GEPA`.
Um run que já estava em andamento carregou o `paper.py` antigo e não gera `tab_floor.tex`; depois que
terminar, rode `--stages paper` de novo.

### 4. Trazer os resultados de volta
```bash
rsync -av user@gpu-host:~/UncertaintyGEPA/paper_materials/myrun/ paper_materials/myrun/
```
e abra `scripts/run.ipynb`.

## Layout
```
paper_materials/
  inputs/llm1/llm1_<tag>_<benchmark>.csv      entrada: rótulo + 20 reamostragens por item
  <experiment>/work/       codebook_<bench>__<cond>.txt, trajectory_*.json, gepa_runs/
  <experiment>/results/    summary_<bench>.csv, per_item_<bench>.csv, floor_*.csv
  <experiment>/paper/      tab_*.tex, macros.tex, codebooks.tex, results_skeleton.tex
  <experiment>/logs/
```
