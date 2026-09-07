"""
Stage 2: unify the evaluation tool. Run the PAPER'S agent-as-a-judge (the 13.py
criteria/schemas/Semantic-Scholar search) on BOTH ours and the strong baseline,
with the model switched to gpt-5.5 (the model the paper text reports).

NOTE: the single-model judge script was run with gpt-4o-mini at the time; this script
re-runs the identical agent logic on gpt-5.5 for both sets so the comparison
uses one consistent evaluator.
"""
# Source file in the working repository: code/16.head2head_agentjudge_gpt55.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import importlib.util as IU
import time, os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

D = _ROOT
MODEL = "gpt-5.5"

# import constants/schemas/search from the paper's judge file (filename has dots/space)
spec = IU.spec_from_file_location("judge13", f"{D}/src/06_evaluation/03_agent_as_a_judge_single_model.py")
J = IU.module_from_spec(spec); spec.loader.exec_module(J)
client = J.client  # OpenAI client w/ the project key
CRITERIA, PERSONA, NOTICE = J.CRITERIA, J.GLOBAL_PERSONA, J.CONCEPT_LEVEL_NOTICE
AgentVerdict, TechCrit, FinalVerdict = J.AgentVerdict, J.TechnicalCriticism, J.FinalTechVerdict
search_fn = J.search_academic_resources


def parse_call(prompt, schema):
    for _ in range(4):
        try:
            r = client.beta.chat.completions.parse(model=MODEL,
                messages=[{"role": "user", "content": prompt}], response_format=schema)
            return r.choices[0].message.parsed
        except Exception:
            time.sleep(3)
    return None


def eval_criterion(text, ckey, ctx):
    p = (f"You are a {PERSONA}.\n{NOTICE}\n\nCriterion: [{ckey}]\nDefinition: {CRITERIA[ckey]}\n\n"
         f"<technology_concept>\n{text}\n</technology_concept>\n\n<search_context>\n{ctx or 'No external data yet.'}</search_context>\n\n"
         "Instructions:\n1. Evaluate whether this CONCEPT logically meets the criterion.\n"
         "2. Judge the idea's theoretical direction, NOT execution maturity.\n"
         "3. Output strict 'YES' or 'NO' (default YES if logic is sound and no clear violation).\n"
         "4. If you must check literature, set needs_more_info=True and give search_queries_used.")
    return parse_call(p, AgentVerdict)


def judge_one(text):
    verdicts = []
    for ckey in CRITERIA.keys():
        v = eval_criterion(text, ckey, "")
        # one reflection round with search, like the paper graph
        if v is not None and v.needs_more_info:
            try:
                qs = v.search_queries_used or [text[:60]]
                ctx = "".join(f"\n[Query:{q}]\n{search_fn(q)}" for q in qs[:2])
            except Exception:
                ctx = ""
            v2 = eval_criterion(text, ckey, ctx)
            v = v2 or v
        verdicts.append(v)
    verdicts = [v for v in verdicts if v is not None]
    yes = sum(1 for v in verdicts if v.verdict == "YES")
    # critic
    cp = (f"You are a Concept-Level Idea Critic.\n{NOTICE}\n<technology_concept>\n{text}\n</technology_concept>\n"
          "Identify ONLY genuine theoretical weaknesses (missing mechanism/data/proto are NOT flaws). "
          "Score conceptual_coherence 1-5 (5=best; absence of mechanism must score >=3 if logic is sound).")
    crit = parse_call(cp, TechCrit)
    coh = crit.conceptual_coherence if crit else -1
    # synthesis
    sp = (f"You are a Meta-Judge.\n{NOTICE}\n=== Summary ===\nTotal YES: {yes}/5\nCoherence: {coh}/5\n"
          f"Flaws: {'; '.join(crit.theoretical_flaws) if crit and crit.theoretical_flaws else 'None'}\n"
          "Rules:\n- HIGH_POTENTIAL: >=4 YES AND coherence>=4\n- MODERATE_POTENTIAL: 2-3 YES OR coherence=3\n"
          "- NEEDS_REVISION: <=1 YES OR coherence<=2\nGive final_synthesis and final_verdict per the rules.")
    fin = parse_call(sp, FinalVerdict)
    return dict(yes=yes, coherence=coh, verdict=fin.final_verdict if fin else "ERROR")


def run_set(texts, name):
    res = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(judge_one, t): i for i, t in enumerate(texts)}
        for f in as_completed(futs):
            res[futs[f]] = f.result()
    ok = [r for r in res if r and r["verdict"] != "ERROR"]
    yes = np.mean([r["yes"] for r in ok]); coh = np.mean([r["coherence"] for r in ok])
    vd = pd.Series([r["verdict"] for r in ok]).value_counts().to_dict()
    high = 100 * np.mean([r["verdict"] == "HIGH_POTENTIAL" for r in ok])
    hm = 100 * np.mean([r["verdict"] in ("HIGH_POTENTIAL", "MODERATE_POTENTIAL") for r in ok])
    print(f"[{name:8s}] n={len(ok)} | mean YES={yes:.2f}/5 | mean coherence={coh:.2f}/5 | "
          f"HIGH={high:.0f}% | HIGH+MOD={hm:.0f}% | verdicts={vd}")
    return dict(name=name, n=len(ok), yes=yes, coherence=coh, high=high, highmod=hm, verdicts=vd)


# ---- load both sets ----
fd = pd.read_csv(f"{D}/data/final_decoded_results_with_nn.csv")
ours = [str(x) for x in fd["decoded_text"].dropna()][:55]
mr = pd.read_csv(f"{D}/data/head2head_multirun.csv")
base = [str(x) for x in mr["baseline_run1"]][:55]

print(f"Agent-as-a-judge unified on {MODEL}; evaluating OURS (n={len(ours)}) and BASELINE (n={len(base)})\n")
ro = run_set(ours, "OURS")
rb = run_set(base, "BASELINE")
print("\n===== UNIFIED AGENT (gpt-5.5) HEAD-TO-HEAD =====")
for k in ["yes", "coherence", "high", "highmod"]:
    print(f"  {k:10s}: OURS={ro[k]:.2f}  BASELINE={rb[k]:.2f}")
pd.DataFrame([ro, rb]).to_csv(f"{D}/data/head2head_agentjudge_gpt55.csv", index=False)
print(_ROOT + "/data/saved head2head_agentjudge_gpt55.csv")
