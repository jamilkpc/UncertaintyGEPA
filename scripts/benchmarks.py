"""Registry de benchmarks. Cada entrada declara como carregar, o prompt do LLM 1, o espaco de
rotulos, como verbalizar o rotulo para o assessor e o construto que abre todo prompt do LLM 2.

Os loaders devolvem um DataFrame com `item_id, text, n_ann, u_m, consensus, split_orig`.
`text` ja vem formatado como o LLM 1 vai ler (ConvAbuse com o contexto do dialogo, AmbiStory
com a narrativa e o sentido candidato).
"""
import json
import urllib.request

import numpy as np
import pandas as pd

import config as C

LEWIDI = ("https://raw.githubusercontent.com/Le-Wi-Di/le-wi-di.github.io/main/"
          "LeWiDi_2-2023/{d}_dataset/{d}_{split}.json")
AMBI = "https://raw.githubusercontent.com/Janosch-Gehring/ambistory/main/{}.json"


def _lewidi(name, task_key=None):
    """task_key=None usa 'annotations'; senao puxa de other_info['other annotations']."""
    rows = []
    for split in ("train", "dev", "test"):
        with urllib.request.urlopen(LEWIDI.format(d=name, split=split)) as r:
            j = json.load(r)
        for k, v in j.items():
            raw = (v["annotations"] if task_key is None
                   else v["other_info"]["other annotations"][task_key])
            xs = raw.split(",")
            if any(x.strip() not in ("0", "1") for x in xs):
                continue                      # 3 itens do offensive trazem "No"
            a = [int(x) for x in xs]
            p = sum(a) / len(a)
            rows.append({"item_id": f"{name}-{split}-{k}", "text": v["text"], "n_ann": len(a),
                         "u_m": p * (1 - p), "consensus": int(p > 0.5), "split_orig": split})
    return pd.DataFrame(rows)


def load_convabuse():
    rows = []
    for split in ("train", "dev", "test"):
        with urllib.request.urlopen(LEWIDI.format(d="ConvAbuse", split=split)) as r:
            j = json.load(r)
        for k, v in j.items():
            a = [int(x) for x in v["annotations"].split(",")]
            p = sum(x == 1 for x in a) / len(a)          # 1 = nao abusivo na escala original
            oi = v.get("other_info", {}) or {}
            ctx = "\n".join(x for x in [
                f"agent: {oi.get('previous_agent_turn','')}".strip(),
                f"user: {oi.get('previous_user_turn','')}".strip(),
                f"agent: {oi.get('agent_turn','')}".strip()] if x.split(": ", 1)[-1])
            rows.append({"item_id": f"ConvAbuse-{split}-{k}",
                         "text": (ctx + "\n" if ctx else "") + f"user: {v['text']}",
                         "n_ann": len(a), "u_m": p * (1 - p),
                         "consensus": int(p <= 0.5),      # abusivo = maioria nao deu 1
                         "split_orig": split})
    return pd.DataFrame(rows)


def load_ambistory():
    rows = []
    for split in ("train", "dev", "test"):
        with urllib.request.urlopen(AMBI.format(split)) as r:
            j = json.load(r)
        for k, v in j.items():
            ch = v["choices"]
            rows.append({
                "item_id": f"AmbiStory-{split}-{k}",
                "text": (f"{v['precontext']}\n{v['sentence']}\n{v['ending']}\n\n"
                         f"AMBIGUOUS WORD: {v['homonym']}\nCANDIDATE MEANING: {v['judged_meaning']}"),
                "n_ann": len(ch), "u_m": float(np.var(ch)),
                "consensus": int(round(np.mean(ch))), "split_orig": split})
    return pd.DataFrame(rows)


def load_mhs(item, n_max=4000):
    from datasets import load_dataset
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", "default")["train"].to_pandas()
    need = {"comment_id", "text", item}
    missing = need - set(ds.columns)
    if missing:
        raise KeyError(f"colunas ausentes no MHS: {missing}. Disponiveis: {sorted(ds.columns)[:40]}")
    g = ds.groupby("comment_id").agg(text=("text", "first"), n_ann=(item, "size"),
                                     u_m=(item, "var"), mean=(item, "mean")).reset_index()
    g = g[g.n_ann >= 2].dropna(subset=["u_m"])
    g = g.sample(min(n_max, len(g)), random_state=C.SEED)
    rng = np.random.default_rng(C.SEED)
    r = rng.random(len(g))
    g["split_orig"] = np.where(r < .7, "train", np.where(r < .85, "dev", "test"))
    g["item_id"] = "MHS" + item + "-" + g.comment_id.astype(str)
    g["consensus"] = g["mean"].round().astype(int)
    return g[["item_id", "text", "n_ann", "u_m", "consensus", "split_orig"]]


def _ann_prompt(unit, construct, labels_desc):
    return (f"You are coding {unit} for a research project.\n\n{construct}\n\n"
            f"Answer with exactly one character: {labels_desc}. Nothing else.")


BENCHMARKS = {
 "HSBrexit": dict(
   loader=lambda: _lewidi("HS-Brexit"), levels=[0,1],
   construct=("An English tweet about Brexit and immigration was labelled for HATE SPEECH: "
              "language that attacks, demeans or excludes people because of their religion, "
              "national origin or migration status."),
   ann=_ann_prompt("English tweets about Brexit and immigration. Placeholders <user> and <url> "
                   "stand for removed mentions and links",
                   "Label the tweet for HATE SPEECH: language that attacks, demeans or excludes "
                   "people because of their religion, national origin or migration status.",
                   "1 (hate speech) or 0 (not hate speech)"),
   verb={0:"not hate speech", 1:"hate speech"}),

 "HSBrexitOff": dict(
   loader=lambda: _lewidi("HS-Brexit", "offensive language detection"), levels=[0,1],
   construct=("An English tweet about Brexit and immigration was labelled for OFFENSIVE LANGUAGE: "
              "insults, slurs, profanity or contempt directed at a person or group."),
   ann=_ann_prompt("English tweets about Brexit and immigration",
                   "Label the tweet for OFFENSIVE LANGUAGE: insults, slurs, profanity or contempt "
                   "directed at a person or group.",
                   "1 (offensive) or 0 (not offensive)"),
   verb={0:"not offensive", 1:"offensive"}),

 "ArMIS": dict(
   loader=lambda: _lewidi("ArMIS"), levels=[0,1],
   construct=("An Arabic tweet was labelled for MISOGYNY OR SEXISM: content that demeans, "
              "stereotypes, polices or dismisses women, or treats them as inferior."),
   ann=_ann_prompt("Arabic tweets (Gulf and Levantine dialects)",
                   "Label the tweet for MISOGYNY OR SEXISM: content that demeans, stereotypes, "
                   "polices or dismisses women, or treats them as inferior.",
                   "1 (misogynistic or sexist) or 0 (not)"),
   verb={0:"not misogynistic", 1:"misogynistic or sexist"}),

 "ConvAbuse": dict(
   loader=load_convabuse, levels=[0,1],
   construct=("The last user turn of a conversation with a chatbot was labelled for ABUSE: "
              "sexist, racist, homophobic, transphobic or ableist language, sexual harassment, "
              "intellectual denigration, or hostility aimed at the system or at a group."),
   ann=_ann_prompt("turns from conversations between a human user and a chatbot",
                   "Label the LAST USER TURN for ABUSE: sexist, racist, homophobic, transphobic "
                   "or ableist language, sexual harassment, intellectual denigration, or hostility "
                   "aimed at the system, at a group, or at an individual.",
                   "1 (abusive) or 0 (not abusive)"),
   verb={0:"not abusive", 1:"abusive"}),

 "MDAgreement": dict(
   loader=lambda: _lewidi("MD-Agreement"), levels=[0,1],
   construct=("An English tweet from one of three domains (Black Lives Matter, the 2020 US "
              "election, Covid-19) was labelled for OFFENSIVENESS."),
   ann=_ann_prompt("English tweets from three domains: Black Lives Matter, the 2020 US election, "
                   "and Covid-19. Placeholders <user> and <url> stand for removed mentions and links",
                   "Label the tweet for OFFENSIVENESS: insults, slurs or profanity aimed at a "
                   "person or group, or language a reader could reasonably take as offensive.",
                   "1 (offensive) or 0 (not offensive)"),
   verb={0:"not offensive", 1:"offensive"}),

 "AmbiStory": dict(
   loader=load_ambistory, levels=[1,2,3,4,5],
   construct=("A short narrative contains an ambiguous word, and a candidate meaning for it was "
              "rated for PLAUSIBILITY on a 1-5 scale, from completely implausible to clearly "
              "intended."),
   ann=_ann_prompt("short narratives containing an ambiguous word",
                   "Rate how plausible it is that the ambiguous word carries the candidate meaning "
                   "in this story, from 1 (completely implausible) to 5 (clearly intended).",
                   "a single digit 1, 2, 3, 4 or 5"),
   verb={i: f"plausibility {i} of 5" for i in range(1, 6)}),

 "MHShatespeech": dict(
   loader=lambda: load_mhs("hatespeech"), levels=[0,1,2],
   construct=("A social media comment was judged for HATE SPEECH: bias-motivated, hostile and "
              "malicious language targeted at a person or group because of their actual or "
              "perceived innate characteristics."),
   ann=_ann_prompt("social media comments from YouTube, Reddit and Twitter",
                   "Does this comment contain hate speech, defined as bias-motivated, hostile and "
                   "malicious language targeted at a person or group because of their actual or "
                   "perceived innate characteristics?",
                   "0 (no), 1 (unclear) or 2 (yes)"),
   verb={0:"not hate speech", 1:"unclear", 2:"hate speech"}),

 "MHSdehumanize": dict(
   loader=lambda: load_mhs("dehumanize"), levels=[0,1,2,3,4],
   construct=("A social media comment was rated on whether it DEHUMANISES the group it refers to, "
              "for example by comparing them to an animal, on a 0-4 agreement scale."),
   ann=_ann_prompt("social media comments from YouTube, Reddit and Twitter",
                   "Rate agreement with: this comment dehumanises the group(s) it refers to (e.g. "
                   "by comparing them to an animal). 0 = strongly disagree, 4 = strongly agree.",
                   "a single digit 0, 1, 2, 3 or 4"),
   verb={i: f"dehumanisation rating {i} of 4" for i in range(5)}),

 "MHSattackdefend": dict(
   loader=lambda: load_mhs("attack_defend"), levels=[0,1,2,3,4],
   construct=("A social media comment was rated on whether it ATTACKS or DEFENDS the group it "
              "refers to, from 0 (strongly defending) to 4 (strongly attacking)."),
   ann=_ann_prompt("social media comments from YouTube, Reddit and Twitter",
                   "Is the comment attacking or defending the group(s) it refers to? "
                   "0 = strongly defending, 4 = strongly attacking.",
                   "a single digit 0, 1, 2, 3 or 4"),
   verb={i: f"attack-defend rating {i} of 4" for i in range(5)}),
}

# benchmarks de um run padrao; acrescente os "MHS*" quando quiser pagar o download e as chamadas
RUN = ["HSBrexit", "ArMIS", "ConvAbuse", "MDAgreement", "AmbiStory"]     # HSBrexitOff continua no registro, fora do escopo
