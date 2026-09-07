# Source file in the working repository: code/17.ablation_text_dropout.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os
"""
Controlled text-dropout ablation (explanation/inference stage only).

Three description sets for the SAME 55 predicted links, evaluated with the
Section-4.5 metric (embed with ada-002, Top-1 cosine to the 120 validation patents):

  A) structure-only  : the 55 vec2text descriptions (FIXED; reads no endpoint
                        text at inference -> invariant baseline)
  B) LLM-full        : an LLM given BOTH endpoints' ABSTRACTS
  C) LLM-title       : the SAME prompt, but evidence = endpoints' TITLES only

Control: instructions are identical across B and C; only the evidence block
changes (abstract -> title), so any gap reflects text availability, not model skill.

Read: B ~= A (parity when text is present); C << A (only the LLM collapses when
text is removed). Paired Wilcoxon on A-vs-C (n=55).
"""
import re, ast, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

D = _ROOT
GEN_MODEL, SEED = "gpt-5.1", 11
K_EMB = os.environ.get("OPENAI_API_KEY")
K_CHAT = os.environ.get("OPENAI_API_KEY")
chat, emb = OpenAI(api_key=K_CHAT), OpenAI(api_key=K_EMB)


def parse(v):
    try:
        return ast.literal_eval(v)
    except Exception:
        return None


pl = pd.read_csv(f"{D}/data/predicted_new_links.csv", dtype=str)
fd = pd.read_csv(f"{D}/data/final_decoded_results_with_nn.csv"); fd["e"] = fd["embedding"].apply(parse)
val = pd.read_csv(f"{D}/data/raw/validation_abstract.csv"); val["e"] = val["embedding"].apply(parse)
ne = pd.read_csv(f"{D}/data/node_embeddings_with_text_embedding_ada.csv", dtype=str)
title_map = {str(r.id): str(r.title) for r in ne.itertuples() if pd.notna(r.title)}
abs_map = {str(r.id): str(r.abstract) for r in ne.itertuples() if pd.notna(r.abstract)}
V = np.array([x for x in val["e"]])
P_struct = np.array([x for x in fd["e"]])
rows = [int(re.search(r"row (\d+)", s).group(1)) for s in fd["node_id"]]
pairs = [(pl.iloc[r]["node1"], pl.iloc[r]["node2"]) for r in rows]

COMMON = ("Two patents are predicted to be co-cited together by a future invention "
          "(a new invention will build on both).\n\n{evidence}\n\n"
          "Describe the prospective invention / technology concept that would jointly build on "
          "both patents. Write 3-4 sentences in patent-abstract style. Output only the description.")
EV_FULL = "PATENT A abstract:\n{a}\n\nPATENT B abstract:\n{b}"
EV_TITLE = "PATENT A title:\n{a}\n\nPATENT B title:\n{b}"


def gen(i, mode):
    a, b = pairs[i]
    if mode == "full":
        ev = EV_FULL.format(a=abs_map.get(a, "")[:1500], b=abs_map.get(b, "")[:1500])
    else:
        ev = EV_TITLE.format(a=title_map.get(a, ""), b=title_map.get(b, ""))
    msg = COMMON.format(evidence=ev)
    for _ in range(4):
        try:
            r = chat.chat.completions.create(model=GEN_MODEL, seed=SEED,
                messages=[{"role": "user", "content": msg}], max_completion_tokens=2200)
            t = (r.choices[0].message.content or "").strip()
            if t:
                return i, t
        except Exception:
            time.sleep(3)
    return i, ""


def gen_set(mode):
    out = [""] * 55
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(gen, i, mode) for i in range(55)]):
            i, t = f.result(); out[i] = t
    return out


def embed(texts):
    r = emb.embeddings.create(model="text-embedding-ada-002", input=[t[:8000] or " " for t in texts])
    return np.array([d.embedding for d in r.data])


def top1(P):
    return cosine_similarity(P, V).max(axis=1)


print("Generating LLM-full ...")
full_txt = gen_set("full")
print("Generating LLM-title ...")
title_txt = gen_set("title")

t_struct = top1(P_struct)
t_full = top1(embed(full_txt))
t_title = top1(embed(title_txt))

print("\n===== Top-1 cosine to 120 validation patents (Section 4.5 metric) =====")
for nm, t in [("A structure-only", t_struct), ("B LLM-full", t_full), ("C LLM-title", t_title)]:
    print(f"  {nm:18s}: mean={t.mean():.4f}  median={np.median(t):.4f}  sd={t.std():.4f}")
print(f"\n  LLM drop  (full -> title): {t_full.mean() - t_title.mean():+.4f}")
print(f"  structure drop           : {0.0:+.4f}  (invariant by construction)")

print("\n===== paired tests (n=55) =====")
for nm, a, b in [("structure vs LLM-title (KEY)", t_struct, t_title),
                 ("LLM-full vs LLM-title", t_full, t_title),
                 ("structure vs LLM-full (parity)", t_struct, t_full)]:
    w, p = stats.wilcoxon(a, b)
    d = (a.mean() - b.mean())
    print(f"  {nm:32s}: mean diff={d:+.4f} | Wilcoxon p={p:.2e}")

pd.DataFrame({"node1": [p[0] for p in pairs], "node2": [p[1] for p in pairs],
              "structure_text": fd["decoded_text"].astype(str), "llm_full": full_txt, "llm_title": title_txt,
              "t1_structure": t_struct, "t1_full": t_full, "t1_title": t_title}).to_csv(
    f"{D}/data/ablation_text_dropout.csv", index=False)
np.savez(f"{D}/data/ablation_text_dropout.npz", struct=t_struct, full=t_full, title=t_title)
print(_ROOT + "/data/\nsaved ablation_text_dropout.csv + .npz")
