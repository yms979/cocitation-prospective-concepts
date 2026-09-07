"""
19. agent-as-a-judge 멀티모델 결과 종합(정직한 버전)
====================================================
18.py 가 저장한 모델별 결과(agentjudge_3model_{gpt,claude,gemini}.xlsx)를 읽어
- 전부 ERROR 인 모델은 UNAVAILABLE(N/A) 로 표시
- 실제 응답한 모델만으로 다수결(consensus / majority) 계산
하여 종합 표(agentjudge_3model_combined.xlsx)를 다시 만든다.
"""
# Source file in the working repository: code/19.agentjudge_combine.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd

D = _ROOT
MODELS = ["GPT", "Claude", "Gemini"]
CRIT_KEYS = [
    "Originality & Novelty", "Scientific & Logical Consistency",
    "Problem–Concept Alignment", "Research & Development Potential",
    "Differentiated Value Proposition",
]

dfs = {m: pd.read_excel(f"{D}/data/agentjudge_3model_{m.lower()}.xlsx") for m in MODELS}

def model_available(df):
    verdicts = [str(df.loc[i, f"{c} | verdict"]) for i in df.index for c in CRIT_KEYS]
    return any(v in ("YES", "NO") for v in verdicts)

avail = {m: model_available(dfs[m]) for m in MODELS}
labels = list(dfs["GPT"]["label"])

# ── overall sheet ──────────────────────────────────────────────
overall = []
for i, lab in enumerate(labels):
    rec = {"label": lab, "node_id": dfs["GPT"].loc[i, "node_id"]}
    votes = []
    for m in MODELS:
        if avail[m]:
            ov = dfs[m].loc[i, "overall_verdict"]
            rec[f"{m}_total_yes"] = int(dfs[m].loc[i, "total_yes"])
            rec[f"{m}_overall"] = ov
            votes.append(ov)
        else:
            rec[f"{m}_total_yes"] = "N/A"
            rec[f"{m}_overall"] = "UNAVAILABLE"
    yc = votes.count("YES")
    rec["available_models"] = ", ".join(m for m in MODELS if avail[m])
    rec["YES_votes"] = f"{yc}/{len(votes)}" if votes else "0/0"
    rec["consensus_overall"] = ("YES" if yc * 2 > len(votes) else "NO") if votes else "N/A"
    overall.append(rec)
overall_df = pd.DataFrame(overall)

# ── by_criterion sheet ─────────────────────────────────────────
by_crit = []
for i, lab in enumerate(labels):
    for c in CRIT_KEYS:
        rec = {"label": lab, "criterion": c}
        votes = []
        for m in MODELS:
            if avail[m]:
                v = str(dfs[m].loc[i, f"{c} | verdict"])
                rec[m] = v
                if v in ("YES", "NO"):
                    votes.append(v)
            else:
                rec[m] = "N/A"
        yc = votes.count("YES")
        rec["majority"] = ("YES" if yc * 2 > len(votes) else "NO") if votes else "N/A"
        by_crit.append(rec)
by_crit_df = pd.DataFrame(by_crit)

out = f"{D}/data/agentjudge_3model_combined.xlsx"
with pd.ExcelWriter(out) as xw:
    overall_df.to_excel(xw, sheet_name="overall", index=False)
    by_crit_df.to_excel(xw, sheet_name="by_criterion", index=False)

print("모델 가용성:", {m: ("OK" if avail[m] else "UNAVAILABLE") for m in MODELS})
print("\n=== OVERALL ===")
print(overall_df.to_string(index=False))
print("\n=== BY CRITERION ===")
print(by_crit_df.to_string(index=False))
print(f"\n저장: {out}")
