"""LLM1 registry extracted verbatim from notebook cells 5–6 (zero based).

No API clients, downstream models, or notebook execution at import time.
The only loader edit is a bounded 60-second URL timeout.
MHS loaders/entries are deliberately excluded at the user's request.
"""
import json
import urllib.request
import numpy as np
import pandas as pd

SEED = 42
R = 20
TEMP_RESAMPLE = 1.0
SOURCE_NOTEBOOK_SHA256 = '5d7bb03c88e69e14e39ec675cfd86bfc6845cfcc8d58c66d5e24ac8ea1cab94b'
LOADER_SOURCE_SHA256 = '8bb68c86f9180a0645accd737510c8601c9c625409d0f84522347b4938e1b69b'

LEWIDI = ("https://raw.githubusercontent.com/Le-Wi-Di/le-wi-di.github.io/main/"
          "LeWiDi_2-2023/{d}_dataset/{d}_{split}.json")

def _lewidi(name, task_key=None):
    """task_key=None usa 'annotations'; senão puxa de other_info['other annotations']."""
    rows = []
    for split in ("train", "dev", "test"):
        with urllib.request.urlopen(LEWIDI.format(d=name, split=split), timeout=60) as r:
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
        with urllib.request.urlopen(LEWIDI.format(d="ConvAbuse", split=split), timeout=60) as r:
            j = json.load(r)
        for k, v in j.items():
            a = [int(x) for x in v["annotations"].split(",")]
            p = sum(x == 1 for x in a) / len(a)          # 1 = não abusivo na escala original
            oi = v.get("other_info", {}) or {}
            ctx = "\n".join(x for x in [
                f"agent: {oi.get('previous_agent_turn','')}".strip(),
                f"user: {oi.get('previous_user_turn','')}".strip(),
                f"agent: {oi.get('agent_turn','')}".strip()] if x.split(": ", 1)[-1])
            rows.append({"item_id": f"ConvAbuse-{split}-{k}",
                         "text": (ctx + "\n" if ctx else "") + f"user: {v['text']}",
                         "n_ann": len(a), "u_m": p * (1 - p),
                         "consensus": int(p <= 0.5),      # abusivo = maioria não deu 1
                         "split_orig": split})
    return pd.DataFrame(rows)

AMBI = "https://raw.githubusercontent.com/Janosch-Gehring/ambistory/main/{}.json"

def load_ambistory():
    rows = []
    for split in ("train", "dev", "test"):
        with urllib.request.urlopen(AMBI.format(split), timeout=60) as r:
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


}

RUN = ["HSBrexit", "ArMIS", "ConvAbuse", "MDAgreement", "AmbiStory"]
DEFAULT_BENCHMARKS = tuple(RUN)
