"""Escreve os .tex do paper em `paper_materials/paper/`. Os numeros do texto entram por macro
(`macros.tex`), entao reexecutar atualiza o paper inteiro sem edicao manual.

`write_all` so precisa de `summary` (results/summary.csv), dos `frames` (texto, g_m, u_m do LLM 1
em cache) e dos codebooks (work/codebook_*.txt). Nenhuma chamada a LLM."""
import glob
import math
import os
import re
import zipfile

import pandas as pd

import config as C
import metrics as M
from benchmarks import RUN

METHOD_ORDER = ["single_hedge", "reclassify", "topk_wrong", "pros_cons", "GEPA"]


def esc(s): return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
def fmt(x, d=3): return "--" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{d}f}"
def mac(name): return "".join(w.capitalize() for w in re.split(r"[^A-Za-z]+", name) if w)
def _write(fname, lines): open(f"{C.PAPERDIR}/{fname}", "w").write("\n".join(lines))


def load_summary():
    """Junta results/summary_<bench>.csv de todos os benchmarks ja avaliados. Cai para o
    results/summary.csv unico dos experimentos antigos."""
    parts = sorted(glob.glob(f"{C.RESDIR}/summary_*.csv"))
    if parts:
        return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    return pd.read_csv(f"{C.RESDIR}/summary.csv")


def load_codebooks():
    out = {}
    for p in sorted(glob.glob(f"{C.WORKDIR}/codebook_*.txt")):
        key = os.path.basename(p)[len("codebook_"):-len(".txt")]
        out[key] = open(p, encoding="utf-8").read()
    return out


def table_data(frames, run):
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lrrrrrr}", r"\hline",
         r"Benchmark & Items & Ann./item & \%\,$u_m{=}0$ & $K_u$ & \%\,$g_m{=}0$ & $\rho(g_m,u_m)$ \\",
         r"\hline"]
    for b in run:
        f = frames[b]; K, _ = M.rho_max_discrete(f.u_m)
        na = f.n_ann.min() if f.n_ann.nunique() == 1 else f"{f.n_ann.min()}--{f.n_ann.max()}"
        L.append(f"{esc(b)} & {len(f)} & {na} & {(f.u_m == 0).mean() * 100:.1f} & {K} & "
                 f"{(f.g_m == 0).mean() * 100:.1f} & {fmt(M.spearman(f.g_m, f.u_m))} \\\\")
    L += [r"\hline", r"\end{tabular}",
          r"\caption{Benchmarks. $u_m$ is human disagreement, $g_m$ the variance of the $R=20$ "
          r"annotator draws, $K_u$ the number of distinct values $u_m$ takes.}",
          r"\label{tab:data}", r"\end{table}"]
    _write("tab_data.tex", L)


def table_main(summary, run, cond, fname, cap):
    """rho_u por benchmark x metodo; melhor por coluna em negrito."""
    piv = (summary[summary.condition == cond].pivot(index="method", columns="benchmark", values="rho_u"))
    order = [m for m in METHOD_ORDER if m in piv.index]
    piv = piv.loc[order, [b for b in run if b in piv.columns]]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{l" + "r" * len(piv.columns) + "}", r"\hline",
         "Method & " + " & ".join(esc(c) for c in piv.columns) + r" \\", r"\hline"]
    for m in piv.index:
        cells = []
        for c in piv.columns:
            v = piv.loc[m, c]
            cells.append((r"\textbf{" + fmt(v) + "}") if v == piv[c].max() else fmt(v))
        L.append(esc(m) + " & " + " & ".join(cells) + r" \\")
    ceil_row = [fmt(summary[(summary.condition == cond) & (summary.benchmark == c)].ceil_u.iloc[0])
                for c in piv.columns]
    L += [r"\hline", r"$\rho_{\max}(H{=}3)$ & " + " & ".join(ceil_row) + r" \\",
          r"\hline", r"\end{tabular}", r"\caption{" + cap + r"}",
          r"\label{tab:main-" + cond + r"}", r"\end{table}"]
    _write(fname, L)


def table_ablation(summary, run):
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llrrr}", r"\hline",
         r"Benchmark & Method & labelled & message-only & $\Delta$ \\", r"\hline"]
    for b in run:
        for m in ["single_hedge", "topk_wrong", "pros_cons", "GEPA"]:
            s = summary[(summary.benchmark == b) & (summary.method == m)]
            if set(s.condition) != {"labelled", "blind"}: continue
            a = s[s.condition == "labelled"].rho_u.iloc[0]; c = s[s.condition == "blind"].rho_u.iloc[0]
            L.append(f"{esc(b)} & {esc(m)} & {fmt(a)} & {fmt(c)} & {fmt(a - c, 3)} \\\\")
    L += [r"\hline", r"\end{tabular}",
          r"\caption{Assessor input ablation. $\Delta>0$ means the assigned label helped.}",
          r"\label{tab:ablation}", r"\end{table}"]
    _write("tab_ablation.tex", L)


def table_stability(summary):
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llrrr}", r"\hline",
         r"Benchmark & Method & Levels & Modal share & Coverage \\", r"\hline"]
    for _, r_ in summary[summary.condition == "blind"].iterrows():
        L.append(f"{esc(r_.benchmark)} & {esc(r_.method)} & {int(r_.levels_used)} & "
                 f"{fmt(r_.modal_share, 2)} & {fmt(r_.coverage, 2)} \\\\")
    L += [r"\hline", r"\end{tabular}",
          r"\caption{Use of the hedge scale. Levels is how many of the $H=3$ hedges a method ever "
          r"emits; modal share is the fraction of items receiving the most common one. A method that "
          r"collapses the scale is capped well below $\rho_{\max}$ regardless of how well it ranks.}",
          r"\label{tab:stability}", r"\end{table}"]
    _write("tab_stability.tex", L)


def load_floor():
    parts = sorted(glob.glob(f"{C.RESDIR}/floor_summary_*.csv"))
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True) if parts else None


def table_floor(floor):
    """Piso de ruido: quanto a mesma avaliacao (mesmo prompt, temperatura 0) varia entre passadas."""
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lllrrrr}", r"\hline",
         r"Benchmark & Condition & Method & Runs & Items changing & $\max|\Delta\rho|$ & 95\% interval \\", r"\hline"]
    for _, r_ in floor.iterrows():
        L.append(f"{esc(r_.benchmark)} & {esc(r_.condition)} & {esc(r_.method)} & {int(r_.k)} & "
                 f"{r_.items_flip * 100:.1f}\\% & {fmt(r_.max_abs_diff)} & "
                 f"[{fmt(r_.diff_ci_lo, 2)}, {fmt(r_.diff_ci_hi, 2)}] \\\\")
    L += [r"\hline", r"\end{tabular}",
          r"\caption{Run-to-run noise of the assessor at temperature 0 on the test split. Items changing is the "
          r"share of items whose hedge differs across identical runs; $\max|\Delta\rho|$ is the largest difference "
          r"in $\rho(h_m,u_m)$ between two runs, with a bootstrap interval over items.}",
          r"\label{tab:floor}", r"\end{table}"]
    _write("tab_floor.tex", L)


def appendix_codebooks(codebooks):
    L = [r"\section{Optimised codebooks}", r"\label{app:codebooks}"]
    for k, cb in sorted(codebooks.items()):
        b, cond = k.split("__")
        L += [r"\subsection*{" + esc(b) + " -- " + esc(cond) + "}", r"\begin{quote}\small",
              esc(cb).replace("\n", "\n\n"), r"\end{quote}"]
    _write("codebooks.tex", L)


def macros(summary, frames, run):
    Mx = [r"% gerado por scripts/paper.py -- nao editar a mao"]
    for _, r_ in summary.iterrows():
        base = mac(r_.benchmark) + mac(r_.condition) + mac(r_.method)
        Mx.append(r"\providecommand{\rhoU" + base + "}{" + fmt(r_.rho_u) + "}")
        Mx.append(r"\providecommand{\ciU" + base + "}{[" + fmt(r_.ci_lo, 2) + ", " + fmt(r_.ci_hi, 2) + "]}")
    for b in run:
        f = frames[b]
        Mx += [r"\providecommand{\n" + mac(b) + "}{" + str(len(f)) + "}",
               r"\providecommand{\accLLMone" + mac(b) + "}{" + fmt((f.llm1_label == f.consensus).mean()) + "}",
               r"\providecommand{\gzero" + mac(b) + "}{" + f"{(f.g_m == 0).mean() * 100:.1f}" + "}",
               r"\providecommand{\rhoGU" + mac(b) + "}{" + fmt(M.spearman(f.g_m, f.u_m)) + "}"]
    Mx += [r"\providecommand{\Rdraws}{" + str(C.R) + "}",
           r"\providecommand{\nHedges}{" + str(C.H) + "}"]
    _write("macros.tex", Mx)


def results_skeleton(run):
    sk = [r"% \input{paper/macros} no preambulo",
          r"\section{Results}", "",
          r"Table~\ref{tab:data} describes the benchmarks. \textbf{[TODO: uma frase sobre a faixa de "
          r"$\rho(g_m,u_m)$ e o que ela implica.]}", "",
          r"\input{paper/tab_data}", "",
          r"Table~\ref{tab:main-blind} reports the headline association. On "
          + esc(run[0]) + r" the optimised codebook reaches $\rho = \rhoU"
          + mac(run[0]) + r"BlindGepa$ (\ciU" + mac(run[0]) + r"BlindGepa), against "
          r"$\rhoU" + mac(run[0]) + r"BlindSingleHedge$ for a single unoptimised hedge. "
          r"\textbf{[TODO: o padrao se mantem nos demais? onde quebra?]}", "",
          r"\input{paper/tab_main}", "",
          r"\paragraph{Assessor input.} \textbf{[TODO: ler a coluna $\Delta$ da Tabela~"
          r"\ref{tab:ablation}. Se $\Delta<0$ de forma consistente, o rotulo atrapalha e a Secao~4 "
          r"precisa ser reescrita nesse ponto.]}", "",
          r"\input{paper/tab_ablation}", "",
          r"\paragraph{Use of the scale.} Table~\ref{tab:stability} reports how much of the "
          r"three-level scale each method actually uses. \textbf{[TODO: comentar se os baselines "
          r"colapsam a escala, e se a vantagem do codebook vem de ordenar melhor ou de usar a "
          r"escala inteira.]}", "",
          r"\input{paper/tab_stability}"]
    _write("results_skeleton.tex", sk)


def write_all(frames, summary=None, codebooks=None, run=None):
    """Escreve todos os .tex. `summary` e `codebooks` caem para o que esta em disco."""
    summary = load_summary() if summary is None else summary
    codebooks = load_codebooks() if codebooks is None else codebooks
    run = run or [b for b in RUN if b in frames]
    table_data(frames, run)
    table_main(summary, run, "blind", "tab_main.tex",
               r"Spearman $\rho(h_m,u_m)$ on held-out test items, message-only condition. "
               r"Best per column in bold.")
    table_ablation(summary, run)
    table_stability(summary)
    appendix_codebooks(codebooks)
    floor = load_floor()
    if floor is not None:
        table_floor(floor)
    macros(summary, frames, run)
    results_skeleton(run)
    written = sorted(os.listdir(C.PAPERDIR))
    print("escritos:", written)
    return written


def show(fname):
    """Devolve o conteudo de um .tex gerado, para ler no notebook."""
    return open(f"{C.PAPERDIR}/{fname}").read()


def zip_materials(path=None):
    """Zipa paper/, results/ e os codebooks (na raiz do repo). No Colab, baixa o arquivo."""
    path = path or str(C.ROOT / "paper_materials.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for root in (C.PAPERDIR, C.RESDIR):
            for dp, _, fs in os.walk(root):
                for f in fs: z.write(os.path.join(dp, f))
        for f in glob.glob(f"{C.WORKDIR}/codebook_*.txt"): z.write(f)
    print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB")
    try:
        from google.colab import files; files.download(path)
    except ImportError:
        print("fora do Colab: zip gravado em", path)
    return path
