"""
Head-to-head: structure->text pipeline (ours)  vs  content-based LLM baseline.

For the SAME 55 predicted links:
  OURS     = vec2text-decoded text from the projected structural embedding
             (uses NO endpoint text, only graph structure)   [already produced]
  BASELINE = an LLM given the TWO endpoint abstracts, asked to describe the
             prospective invention bridging them              [uses full content]

Both are compared on identical metrics:
  - validation Top-1 cosine (resemblance to real future patents)
  - peakedness  (Top-1 - mean similarity to all validation docs)  = specificity
  - target diversity (distinct Top-1 targets; internal pairwise cosine)
  - a single BLINDED LLM judge (same prompt/model) -> novelty/coherence/feasibility + PROMISING
"""
# Source file in the working repository: code/14.head2head_content_baseline.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os, re, ast, json, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

D = _ROOT
GEN_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
K_EMB = os.environ.get("OPENAI_API_KEY")
K_CHAT = os.environ.get("OPENAI_API_KEY")
chat = OpenAI(api_key=K_CHAT)
emb = OpenAI(api_key=K_EMB)


def parse(v):
    try:
        return ast.literal_eval(v)
    except Exception:
        return None


# ---------- data ----------
pl = pd.read_csv(f"{D}/data/predicted_new_links.csv", dtype=str)
fd = pd.read_csv(f"{D}/data/final_decoded_results_with_nn.csv")
fd["e"] = fd["embedding"].apply(parse)
val = pd.read_csv(f"{D}/data/raw/validation_abstract.csv")
val["e"] = val["embedding"].apply(parse)
ne = pd.read_csv(f"{D}/data/node_embeddings_with_text_embedding_ada.csv", dtype=str)
abs_map = {str(r.id): str(r.abstract) for r in ne.itertuples() if pd.notna(r.abstract)}
V = np.array([x for x in val["e"]])
P_ours = np.array([x for x in fd["e"]])
rows = [int(re.search(r"row (\d+)", s).group(1)) for s in fd["node_id"]]
pairs = [(pl.iloc[r]["node1"], pl.iloc[r]["node2"]) for r in rows]

# ---------- 1. generate content baseline ----------
GEN_PROMPT = (
    "Two patents are predicted to be co-cited together by a future invention "
    "(i.e., a new invention will build on both).\n\n"
    "PATENT A abstract:\n{a}\n\nPATENT B abstract:\n{b}\n\n"
    "Describe the prospective invention / technology concept that would jointly "
    "build on both patents. Write 3-4 sentences in the style of a patent abstract. "
    "Output only the description, no preamble."
)


def gen_one(i):
    a, b = pairs[i]
    msg = GEN_PROMPT.format(a=abs_map.get(a, "")[:1500], b=abs_map.get(b, "")[:1500])
    for _ in range(4):
        try:
            r = chat.chat.completions.create(model=GEN_MODEL,
                messages=[{"role": "user", "content": msg}], temperature=0.7, max_tokens=220)
            return i, r.choices[0].message.content.strip()
        except Exception as e:
            time.sleep(2)
    return i, ""


print("1. Generating content baseline (55 links)...")
base_texts = [""] * 55
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(gen_one, i) for i in range(55)]
    for f in as_completed(futs):
        i, t = f.result()
        base_texts[i] = t
print("   done." if all(base_texts) else f"   {sum(1 for t in base_texts if not t)} empty")


# ---------- 2. embed baseline ----------
def embed(texts):
    out = []
    for k in range(0, len(texts), 64):
        r = emb.embeddings.create(model="text-embedding-ada-002", input=[t[:8000] for t in texts[k:k+64]])
        out += [d.embedding for d in r.data]
    return np.array(out)


print("2. Embedding baseline...")
P_base = embed(base_texts)


# ---------- 3. metrics ----------
def metrics(P, name):
    S = cosine_similarity(P, V)
    t1 = S.max(axis=1)
    peak = (S.max(axis=1) - S.mean(axis=1)).mean()
    tgt = S.argmax(axis=1)
    distinct = len(set(tgt.tolist()))
    G = cosine_similarity(P)
    intra = (G.sum() - len(P)) / (len(P) ** 2 - len(P))
    print(f"  [{name:8s}] Top-1 mean={t1.mean():.4f} median={np.median(t1):.4f} | "
          f"peakedness={peak:.4f} | distinct targets={distinct}/55 | internal cos={intra:.4f}")
    return dict(name=name, top1=t1.mean(), peak=peak, distinct=distinct, intra=intra, t1arr=t1.tolist())


print("3. Embedding metrics (vs validation):")
m_ours = metrics(P_ours, "OURS")
m_base = metrics(P_base, "BASELINE")
# real-patent reference peakedness ~0.079 (computed earlier)
print("  (real-patent reference: peakedness~0.079, internal cos~0.788)")


# ---------- 4. blinded identical judge ----------
JUDGE_SYS = (
    "You are evaluating an early-stage TECHNOLOGY CONCEPT (not a finished invention). "
    "Absence of detailed mechanism/data is normal and must NOT be penalized. "
    "Rate the concept and return STRICT JSON with integer 1-5 fields "
    '{"novelty":, "coherence":, "feasibility":, "verdict":"PROMISING"|"NOT_PROMISING"}.'
)


def judge_one(text):
    for _ in range(4):
        try:
            r = chat.chat.completions.create(model=JUDGE_MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": JUDGE_SYS},
                          {"role": "user", "content": f"Concept description:\n{text[:1500]}"}])
            return json.loads(r.choices[0].message.content)
        except Exception:
            time.sleep(2)
    return None


def judge_set(texts, name):
    res = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(judge_one, t): i for i, t in enumerate(texts)}
        for f in as_completed(futs):
            res[futs[f]] = f.result()
    res = [r for r in res if r]
    nov = np.mean([r["novelty"] for r in res]); coh = np.mean([r["coherence"] for r in res])
    fea = np.mean([r["feasibility"] for r in res])
    prom = 100 * np.mean([r["verdict"] == "PROMISING" for r in res])
    print(f"  [{name:8s}] novelty={nov:.2f} coherence={coh:.2f} feasibility={fea:.2f} | PROMISING={prom:.0f}% (n={len(res)})")
    return dict(name=name, novelty=nov, coherence=coh, feasibility=fea, promising=prom)


print("4. Blinded identical judge (same prompt/model on both):")
j_ours = judge_set(list(fd["decoded_text"].astype(str)), "OURS")
j_base = judge_set(base_texts, "BASELINE")

# ---------- save ----------
pd.DataFrame({"link_row": rows, "node1": [p[0] for p in pairs], "node2": [p[1] for p in pairs],
              "ours_text": fd["decoded_text"].astype(str), "baseline_text": base_texts}).to_csv(
    f"{D}/data/head2head_descriptions.csv", index=False)
print(_ROOT + "/data/\nsaved head2head_descriptions.csv")
print("\n===== SUMMARY (ours = structure-only;  baseline = content/abstracts) =====")
for k in ["top1", "peak", "distinct", "intra"]:
    print(f"  {k:9s}: OURS={m_ours[k]:.4f}  BASELINE={m_base[k]:.4f}")
for k in ["novelty", "coherence", "feasibility", "promising"]:
    print(f"  {k:11s}: OURS={j_ours[k]:.2f}  BASELINE={j_base[k]:.2f}")
