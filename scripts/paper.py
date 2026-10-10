"""Escreve os .tex do paper em `<experimento>/paper/`. Os numeros do texto entram por macro
(`macros.tex`), entao reexecutar atualiza o paper inteiro sem edicao manual.

As tabelas sao SEMPRE reconstruidas a partir de tudo o que existe em disco para o experimento (todos os
benchmarks com `results/summary_<bench>.csv`), e nao so dos benchmarks da execucao atual: rodar um
benchmark novo acrescenta uma coluna/linha, nunca apaga os outros. Nenhuma chamada a LLM."""
import glob
import math
import os
import re
import zipfile

import pandas as pd

import config as C
from benchmarks import RUN

METHOD_ORDER = ["single_hedge", "reclassify", "topk_wrong", "pros_cons", "GEPA"]


def esc(s): return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
def fmt(x, d=3): return "--" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{d}f}"
def sfmt(x, d=3): return "--" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:+.{d}f}"
def mac(name): return "".join(w.capitalize() for w in re.split(r"[^A-Za-z]+", name) if w)
def _write(fname, lines): open(f"{C.PAPERDIR}/{fname}", "w").write("\n".join(lines))
def _nan(x): return x is None or (isinstance(x, float) and math.isnan(x))


# ---- leitura do que existe em disco -----------------------------------------------------------
def benchmarks_on_disk():
    """Benchmarks com summary no experimento, na ordem padrao de RUN."""
    found = {os.path.basename(p)[len("summary_"):-4] for p in glob.glob(f"{C.RESDIR}/summary_*.csv")}
    return [b for b in RUN if b in found] + sorted(found - set(RUN))


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


def load_floor():
    parts = sorted(glob.glob(f"{C.RESDIR}/floor_summary_*.csv"))
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True) if parts else None


def load_data_stats(run, frames=None):
    """results/data_<bench>.csv; se faltar e houver `frames`, calcula e grava."""
    import evaluation
    rows = []
    for b in run:
        p = f"{C.RESDIR}/data_{b}.csv"
        if os.path.exists(p):
            rows.append(pd.read_csv(p).iloc[0].to_dict())
        elif frames is not None and b in frames:
            rows.append(evaluation.data_stats(frames, b))
        else:
            print(f"aviso: sem estatisticas de dados para {b} (rode o estagio paper com --llm1-tag para gera-las)")
    return pd.DataFrame(rows)


def load_paired(run):
    import evaluation
    parts = [evaluation.ensure_paired(b) for b in run if os.path.exists(f"{C.RESDIR}/per_item_{b}.csv")]
    return pd.concat(parts, ignore_index=True) if parts else None


# ---- tabelas ------------------------------------------------------------------------------------
def table_data(stats):
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{lrrrrrr}", r"\hline",
         r"Benchmark & Items & Ann./item & \%\,$u_m{=}0$ & $K_u$ & \%\,$g_m{=}0$ & $\rho(g_m,u_m)$ \\",
         r"\hline"]
    for _, r in stats.iterrows():
        na = int(r.ann_min) if r.ann_min == r.ann_max else f"{int(r.ann_min)}--{int(r.ann_max)}"
        L.append(f"{esc(r.benchmark)} & {int(r['items'])} & {na} & {r.pct_um0:.1f} & {int(r.K_u)} & "
                 f"{r.pct_gm0:.1f} & {fmt(r.rho_gu)} \\\\")
    L += [r"\hline", r"\end{tabular}",
          r"\caption{Benchmarks. $u_m$ is human disagreement, $g_m$ the variance of the $R=20$ "
          r"annotator draws, $K_u$ the number of distinct values $u_m$ takes.}",
          r"\label{tab:data}", r"\end{table}"]
    _write("tab_data.tex", L)


def _cell(r, best):
    if r is None or _nan(r.rho_u): return "--"
    s = fmt(r.rho_u)
    if best: s = r"\textbf{" + s + "}"
    if not _nan(r.ci_lo): s += r"{\scriptsize\,[" + fmt(r.ci_lo, 2) + ", " + fmt(r.ci_hi, 2) + "]}"
    if r.levels_used < C.H: s += r"$^\dagger$"
    return s


def table_main(summary, run):
    """rho(h, u) no teste, por metodo e benchmark, com intervalo de 95%: painel com rotulo e painel so com
    a mensagem. Melhor por coluna e painel em negrito; † marca metodo que emitiu menos que os H hedges."""
    cols = [b for b in run if b in set(summary.benchmark)]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\hline",
         "Method & " + " & ".join(esc(c) for c in cols) + r" \\", r"\hline"]
    for cond, title in (("labelled", "Label-conditioned"), ("blind", "Message-only")):
        sub = summary[summary.condition == cond]
        if sub.empty: continue
        L.append(r"\multicolumn{%d}{l}{\emph{%s}} \\" % (len(cols) + 1, title))
        for m in [m for m in METHOD_ORDER if m in set(sub.method)]:
            cells = []
            for b in cols:
                r = sub[(sub.benchmark == b) & (sub.method == m)]
                r = r.iloc[0] if len(r) else None
                best = r is not None and not _nan(r.rho_u) and r.rho_u == sub[sub.benchmark == b].rho_u.max()
                cells.append(_cell(r, best))
            L.append(esc(m) + " & " + " & ".join(cells) + r" \\")
        ceil = [fmt(sub[sub.benchmark == b].ceil_u.iloc[0]) if (sub.benchmark == b).any() else "--" for b in cols]
        L += [r"$\rho_{\max}(H{=}3)$ & " + " & ".join(ceil) + r" \\", r"\hline"]
    L += [r"\end{tabular}",
          r"\caption{Spearman $\rho(h_m,u_m)$ between the emitted hedge and human disagreement on the held-out test "
          r"split, with 95\% bootstrap intervals. Best per column and panel in bold; $^\dagger$ marks a method that "
          r"emitted fewer than three hedges.}",
          r"\label{tab:main}", r"\end{table}"]
    _write("tab_main.tex", L)


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


def table_stability(summary, run):
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llrrr}", r"\hline",
         r"Benchmark & Method & Levels & Modal share & Coverage \\", r"\hline"]
    s = summary[summary.condition == "blind"]
    for b in run:
        for _, r_ in s[s.benchmark == b].iterrows():
            L.append(f"{esc(r_.benchmark)} & {esc(r_.method)} & {int(r_.levels_used)} & "
                     f"{fmt(r_.modal_share, 2)} & {fmt(r_.coverage, 2)} \\\\")
    L += [r"\hline", r"\end{tabular}",
          r"\caption{Use of the hedge scale. Levels is how many of the $H=3$ hedges a method ever "
          r"emits; modal share is the fraction of items receiving the most common one. A method that "
          r"collapses the scale is capped well below $\rho_{\max}$ regardless of how well it ranks.}",
          r"\label{tab:stability}", r"\end{table}"]
    _write("tab_stability.tex", L)


def _dcell(r):
    if r is None or len(r) == 0: return "--"
    r = r.iloc[0]
    return f"{sfmt(r.delta)}" + r"{\scriptsize\,[" + sfmt(r.ci_lo, 2) + ", " + sfmt(r.ci_hi, 2) + "]}"


def table_paired(paired, summary, run):
    """Diferencas pareadas de rho(h, u) nos mesmos itens de teste, com intervalo bootstrap sobre os itens:
    codebook menos hedge unico (por condicao), hedge menos o proprio alvo g_m (na condicao em que o codebook foi
    melhor) e rotulo menos sem rotulo (codebook)."""
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lrrrr}", r"\hline",
         r"Benchmark & \multicolumn{2}{c}{codebook $-$ single hedge} & hedge $-$ $g_m$ & label $-$ no label \\",
         r" & message-only & label-cond. & (as predictors of $u_m$) & (codebook) \\", r"\hline"]
    for b in run:
        P = paired[paired.benchmark == b]
        if P.empty: continue
        g = summary[(summary.benchmark == b) & (summary.method == "GEPA")]
        best = g.sort_values("rho_u").condition.iloc[-1] if len(g) else "blind"
        c1 = P[(P.kind == "method_vs_seed") & (P.condition == "blind") & (P.method == "GEPA")]
        c2 = P[(P.kind == "method_vs_seed") & (P.condition == "labelled") & (P.method == "GEPA")]
        c3 = P[(P.kind == "hedge_vs_g") & (P.condition == best)]
        c4 = P[(P.kind == "label_effect") & (P.method == "GEPA")]
        L.append(f"{esc(b)} & {_dcell(c1)} & {_dcell(c2)} & {_dcell(c3)} & {_dcell(c4)} \\\\")
    L += [r"\hline", r"\end{tabular}",
          r"\caption{Paired differences in $\rho(\cdot,u_m)$, bootstrapped over test items (4,000 resamples, common "
          r"seed), with 95\% percentile intervals; positive favours the first term. The third column compares the "
          r"hedge with the instability target it was trained on, in whichever condition the codebook scored higher.}",
          r"\label{tab:paired}", r"\end{table}"]
    _write("tab_paired.tex", L)


def table_paired_methods(paired, run):
    """Todos os metodos contra o hedge unico (comparador fixado de antemao), por benchmark e condicao."""
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lllr}", r"\hline",
         r"Benchmark & Condition & Method & $\Delta\rho(h_m,u_m)$ vs single hedge \\", r"\hline"]
    for b in run:
        P = paired[(paired.benchmark == b) & (paired.kind == "method_vs_seed")]
        for cond in ("blind", "labelled"):
            for m in [m for m in METHOD_ORDER if m in set(P.method)]:
                r = P[(P.condition == cond) & (P.method == m)]
                if len(r): L.append(f"{esc(b)} & {esc(cond)} & {esc(m)} & {_dcell(r)} \\\\")
    L += [r"\hline", r"\end{tabular}",
          r"\caption{Each method against the single unoptimised hedge, paired over test items, with 95\% intervals.}",
          r"\label{tab:paired-methods}", r"\end{table}"]
    _write("tab_paired_methods.tex", L)


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


def macros(summary, stats, paired):
    Mx = [r"% gerado por scripts/paper.py -- nao editar a mao"]
    for _, r_ in summary.iterrows():
        base = mac(r_.benchmark) + mac(r_.condition) + mac(r_.method)
        Mx.append(r"\providecommand{\rhoU" + base + "}{" + fmt(r_.rho_u) + "}")
        Mx.append(r"\providecommand{\ciU" + base + "}{[" + fmt(r_.ci_lo, 2) + ", " + fmt(r_.ci_hi, 2) + "]}")
    for _, r_ in stats.iterrows():
        b = mac(r_.benchmark)
        Mx += [r"\providecommand{\n" + b + "}{" + str(int(r_["items"])) + "}",
               r"\providecommand{\accLLMone" + b + "}{" + fmt(r_.acc_llm1) + "}",
               r"\providecommand{\gzero" + b + "}{" + f"{r_.pct_gm0:.1f}" + "}",
               r"\providecommand{\rhoGU" + b + "}{" + fmt(r_.rho_gu) + "}"]
    if paired is not None:
        for _, r_ in paired[(paired.kind == "method_vs_seed") & (paired.method == "GEPA")].iterrows():
            base = mac(r_.benchmark) + mac(r_.condition)
            Mx.append(r"\providecommand{\dGepaSeed" + base + "}{" + sfmt(r_.delta) + "}")
            Mx.append(r"\providecommand{\ciGepaSeed" + base + "}{[" + sfmt(r_.ci_lo, 2) + ", " + sfmt(r_.ci_hi, 2) + "]}")
    Mx += [r"\providecommand{\Rdraws}{" + str(C.R) + "}", r"\providecommand{\nHedges}{" + str(C.H) + "}"]
    _write("macros.tex", Mx)


def results_skeleton(run):
    b = mac(run[0])
    sk = [r"% \input{paper/macros} no preambulo",
          r"\section{Results}", "",
          r"Table~\ref{tab:data} describes the benchmarks. \textbf{[TODO: uma frase sobre a faixa de "
          r"$\rho(g_m,u_m)$ e o que ela implica.]}", "",
          r"\input{paper/tab_data}", "",
          r"Table~\ref{tab:main} reports the headline association. On " + esc(run[0])
          + r" the optimised codebook reaches $\rho = \rhoU" + b + r"BlindGepa$ (\ciU" + b + r"BlindGepa) in the "
          r"message-only condition, against $\rhoU" + b + r"BlindSingleHedge$ for a single unoptimised hedge. "
          r"\textbf{[TODO: o padrao se mantem nos demais? onde quebra?]}", "",
          r"\input{paper/tab_main}", "",
          r"Table~\ref{tab:paired} gives the paired differences. \textbf{[TODO: ler as colunas e os intervalos.]}", "",
          r"\input{paper/tab_paired}", "",
          r"\paragraph{Assessor input.} \textbf{[TODO: ler a coluna $\Delta$ da Tabela~\ref{tab:ablation} e a "
          r"ultima coluna da Tabela~\ref{tab:paired}.]}", "",
          r"\input{paper/tab_ablation}", "",
          r"\paragraph{Use of the scale.} Table~\ref{tab:stability} reports how much of the "
          r"three-level scale each method actually uses. \textbf{[TODO: comentar se os baselines "
          r"colapsam a escala.]}", "",
          r"\input{paper/tab_stability}"]
    _write("results_skeleton.tex", sk)


def load_selected():
    parts = sorted(glob.glob(f"{C.RESDIR}/selected_*.csv"))
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True) if parts else None


def table_selected(sel, run):
    """A condicao como hiperparametro: por metodo, a condicao com maior rho(h, u) no dev, e no teste dessa condicao
    rho(h, u) (com intervalo) e rho(h, g), o objetivo do GEPA (L = com rotulo, B = so a mensagem). Melhor por
    coluna em negrito."""
    cols = [b for b in run if b in set(sel.benchmark)]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{l" + "rr" * len(cols) + "}", r"\hline",
         " & " + " & ".join(r"\multicolumn{2}{c}{" + esc(c) + "}" for c in cols) + r" \\",
         "Method & " + " & ".join([r"$\rho(h,u)$ & $\rho(h,g)$"] * len(cols)) + r" \\", r"\hline"]
    for m in [m for m in METHOD_ORDER if m in set(sel.method)]:
        cells = []
        for b in cols:
            r = sel[(sel.benchmark == b) & (sel.method == m)]
            if r.empty: cells += ["--", "--"]; continue
            r = r.iloc[0]; sb = sel[sel.benchmark == b]
            cells.append(_cell(r, r.rho_u == sb.rho_u.max())
                         + r"{\scriptsize\,(" + ("L" if r.condition == "labelled" else "B") + ")}")
            g = fmt(r.rho_g)
            cells.append(r"\textbf{" + g + "}" if r.rho_g == sb.rho_g.max() else g)
        L.append(esc(m) + " & " + " & ".join(cells) + r" \\")
    L += [r"\hline", r"\end{tabular}",
          r"\caption{Spearman correlations on the held-out test split, with the assessor input treated as a "
          r"hyperparameter: for each method and benchmark, the condition (L: label-conditioned, B: message-only) "
          r"with the higher $\rho(h_m,u_m)$ on the development split. $\rho(h,u)$ is the association with human "
          r"disagreement, with 95\% bootstrap intervals; $\rho(h,g)$ is the association with the annotator's "
          r"instability target, the objective the master optimised. Best per column in bold; $^\dagger$ marks a "
          r"method that emitted fewer than three hedges.}",
          r"\label{tab:selected}", r"\end{table}"]
    _write("tab_selected.tex", L)


def write_all(frames=None, summary=None, codebooks=None, run=None):
    """Escreve todos os .tex a partir do que existe em disco para o experimento (todos os benchmarks).
    `frames` so e usado para gerar estatisticas de dados que ainda nao estejam em results/data_<bench>.csv."""
    summary = load_summary() if summary is None else summary
    codebooks = load_codebooks() if codebooks is None else codebooks
    run = run or benchmarks_on_disk()
    stats = load_data_stats(run, frames)
    paired = load_paired(run)
    table_data(stats)
    table_main(summary, run)
    sel = load_selected()
    if sel is not None:
        table_selected(sel, run)
    if {"blind", "labelled"} <= set(summary.condition):      # a ablacao precisa das duas condicoes
        table_ablation(summary, run)
    table_stability(summary, run)
    if paired is not None:
        table_paired(paired, summary, run)
        table_paired_methods(paired, run)
    appendix_codebooks(codebooks)
    floor = load_floor()
    if floor is not None:
        table_floor(floor)
    macros(summary, stats, paired)
    results_skeleton(run)
    written = sorted(os.listdir(C.PAPERDIR))
    print("benchmarks:", run, "\nescritos:", written)
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
