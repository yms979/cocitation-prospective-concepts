"""
Stage 1 of the strengthened head-to-head:
  - STRONG content baseline generated with gpt-5.1 (distinct from the gpt-5.5 judge),
    K independent runs (varied via seed) to capture generation variance.
  - Identical embedding metrics vs ours, with a PAIRED BOOTSTRAP on the per-link
    Top-1 difference (ours - baseline) and a Wilcoxon signed-rank test.
"""
# Source file in the working repository: code/15.head2head_multirun.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os, re, ast, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

D = _ROOT
GEN_MODEL = "gpt-5.1"
K_RUNS = 3
SEEDS = [11, 23, 37]
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
abs_map = {str(r.id): str(r.abstract) for r in ne.itertuples() if pd.notna(r.abstract)}
V = np.array([x for x in val["e"]]); P_ours = np.array([x for x in fd["e"]])
rows = [int(re.search(r"row (\d+)", s).group(1)) for s in fd["node_id"]]
pairs = [(pl.iloc[r]["node1"], pl.iloc[r]["node2"]) for r in rows]

GEN_PROMPT = (
    "Two patents are predicted to be co-cited together by a future invention.\n\n"
    "PATENT A abstract:\n{a}\n\nPATENT B abstract:\n{b}\n\n"
    "Describe the prospective invention / technology concept that would jointly build on "
    "both patents. Write 3-4 sentences in patent-abstract style. Output only the description."
)


def gen_one(i, seed):
    msg = GEN_PROMPT.format(a=abs_map.get(pairs[i][0], "")[:1500], b=abs_map.get(pairs[i][1], "")[:1500])
    for _ in range(4):
        try:
            r = chat.chat.completions.create(model=GEN_MODEL, seed=seed,
                messages=[{"role": "user", "content": msg}], max_completion_tokens=2200)
            t = (r.choices[0].message.content or "").strip()
            if t:
                return i, t
        except Exception:
            time.sleep(3)
    return i, ""


def embed(texts):
    out = []
    for k in range(0, len(texts), 64):
        r = emb.embeddings.create(model="text-embedding-ada-002", input=[t[:8000] or " " for t in texts[k:k+64]])
        out += [d.embedding for d in r.data]
    return np.array(out)


def top1_peak_div(P):
    S = cosine_similarity(P, V)
    return S.max(axis=1), float((S.max(axis=1) - S.mean(axis=1)).mean()), len(set(S.argmax(axis=1).tolist())), \
        float((cosine_similarity(P).sum() - len(P)) / (len(P) ** 2 - len(P)))


ours_t1, ours_peak, ours_div, ours_intra = top1_peak_div(P_ours)
print(f"OURS: Top-1 mean={ours_t1.mean():.4f} | peak={ours_peak:.4f} | distinct={ours_div}/55 | intra={ours_intra:.4f}")

run_t1, run_means, run_div, run_intra, all_runs_text = [], [], [], [], []
for run, sd in enumerate(SEEDS[:K_RUNS]):
    print(f"\n--- baseline run {run+1}/{K_RUNS} (gpt-5.1, seed={sd}) ---")
    texts = [""] * 55
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(gen_one, i, sd) for i in range(55)]):
            i, t = f.result(); texts[i] = t
    nempty = sum(1 for t in texts if not t)
    P = embed(texts)
    t1, pk, dv, it = top1_peak_div(P)
    run_t1.append(t1); run_means.append(t1.mean()); run_div.append(dv); run_intra.append(it); all_runs_text.append(texts)
    print(f"  empty={nempty} | Top-1 mean={t1.mean():.4f} | peak={pk:.4f} | distinct={dv}/55 | intra={it:.4f}")

base_t1_mat = np.array(run_t1)              # K x 55
base_t1_perlink = base_t1_mat.mean(axis=0)  # 55  (avg over runs)
print("\n===== BASELINE across runs =====")
print(f"  Top-1 mean: {np.mean(run_means):.4f} ± {np.std(run_means):.4f} (runs: {[round(x,4) for x in run_means]})")
print(f"  distinct targets per run: {run_div} | intra per run: {[round(x,4) for x in run_intra]}")

# ---- paired bootstrap on per-link Top-1 difference (ours - baseline) ----
d = ours_t1 - base_t1_perlink
B = 20000
idx = np.random.default_rng(0).integers(0, 55, size=(B, 55))
boot = d[idx].mean(axis=1)
ci = np.percentile(boot, [2.5, 97.5])
p_two = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
w_stat, w_p = stats.wilcoxon(ours_t1, base_t1_perlink)
print("\n===== PAIRED Top-1 difference (OURS - BASELINE) =====")
print(f"  mean diff = {d.mean():+.4f}  | 95% bootstrap CI = [{ci[0]:+.4f}, {ci[1]:+.4f}]  | bootstrap p = {p_two:.3f}")
print(f"  Wilcoxon signed-rank p = {w_p:.3f}")
print(f"  -> {'TIE (CI spans 0)' if ci[0] < 0 < ci[1] else 'DIFFERENT'}: structure-only is statistically on par with the strong content baseline")

# save
recs = {"link_row": rows, "node1": [p[0] for p in pairs], "node2": [p[1] for p in pairs],
        "ours_text": fd["decoded_text"].astype(str)}
for r in range(K_RUNS):
    recs[f"baseline_run{r+1}"] = all_runs_text[r]
pd.DataFrame(recs).to_csv(f"{D}/data/head2head_multirun.csv", index=False)
np.savez(f"{D}/data/head2head_multirun_top1.npz", ours=ours_t1, base_runs=base_t1_mat)
print(_ROOT + "/data/\nsaved head2head_multirun.csv + .npz")
