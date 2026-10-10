"""Prompts do assessor: semente, baselines (adaptados de Xiong et al. 2024) e o template de reflexao."""
import config as C
from benchmarks import BENCHMARKS


def context(bench):
    """Construto e rotulos que o LLM 1 podia dar, na forma verbal que o assessor le. Abre a semente e
    todos os baselines, para que todos os metodos partam da mesma informacao."""
    b = BENCHMARKS[bench]
    labs = "\n".join(f"- {b['verb'][l]}" for l in b["levels"])
    return f"{b['construct']}\nThe annotator could assign only these labels:\n{labs}\n\n"


def seed_codebook(bench, cond):
    return context(bench) + ("Judge how uncertain that label is for this item." if cond == "labelled"
                             else "Judge how uncertain a coder deciding this item's label would be.")


def baselines(bench, cond):
    c = context(bench)
    B = {"single_hedge": seed_codebook(bench, cond)}
    if cond == "labelled":
        B["reclassify"] = (c + "First decide the label yourself and justify it in one line. Then "
                           "compare your conclusion with the assigned label; the harder the "
                           "disagreement is to settle, the more uncertain the label.")
        B["topk_wrong"] = (c + "List up to three reasons why the assigned label could be WRONG for "
                           "this item, then rate the uncertainty of the label in light of them.")
        B["pros_cons"]  = (c + "List up to three reasons FOR the label being right and up to three "
                           "AGAINST, then weigh them.")
    else:
        B["topk_wrong"] = (c + "List up to three readings of this item a competent coder could "
                           "defend, then rate how uncertain the decision would be.")
        B["pros_cons"]  = (c + "List up to three reasons FOR and three AGAINST the positive label, "
                           "then weigh them.")
    return B


REFLECTION_TEMPLATE = (
    f"The assistant ends its answer with exactly one of these {C.H} uncertainty levels, from least to "
    "most uncertain:\n" + "\n".join(f'- "{h}"' for h in C.HEDGES) + "\n"
    "This scale is fixed. The codebook must use these exact names and no others.\n\n"
    "An assistant was given the instructions below to judge how uncertain an annotation is:\n"
    "---\n<curr_param>\n---\n\n"
    "Below are items, what the assistant answered, and feedback on where it went wrong:\n"
    "---\n<side_info>\n---\n\n"
    "Rewrite the instructions so the assistant judges uncertainty better. Use the examples and the "
    "feedback to work out for yourself what makes an annotation more or less uncertain in this kind "
    "of text. Write it as an annotation codebook: name the dimensions of the construct, give "
    "observable indicators for each, and state how they combine. Form and structure are otherwise "
    "free.\n\nDo not write the final output instruction: it is fixed and appended after your text. "
    "Give only the new instructions, between the --- lines.")
