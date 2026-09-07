#!/usr/bin/env python3
"""Build the validation tables of the manuscript from the released evaluation files.

Table 4  Concept quality assessment: passes per criterion and per judge (n = 55).
Table 5  Distribution of Top-1 cosine similarity between the unanimously approved
         concepts and their most similar subsequent patents (n = 36).

Inputs   data/agentjudge_3model_{gpt,claude,gemini}.xlsx
         data/agentjudge_3model_combined.xlsx
         data/similarity_original_vs_corrected_2024plus.xlsx
Outputs  tables/table4_concept_quality.{csv,xlsx}
         tables/table5_similarity_distribution.{csv,xlsx}
"""
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
D = os.path.join(ROOT, "data")
T = os.path.join(ROOT, "tables")

JUDGES = {"GPT": "gpt-5.4-mini", "Claude": "claude-opus-4.8", "Gemini": "gemini-3.5-flash"}
CRITERIA = {"Novelty": "Novelty & Differentiation",
            "Plausibility": "Scientific & Logical Soundness",
            "Significance": "Significance & Value"}
N = 55


def pct(k, n):
    return f"{k} ({k / n * 100:.1f}%)"


def table4():
    rows = []
    per_model = {m: pd.read_excel(os.path.join(D, f"agentjudge_3model_{m.lower()}.xlsx")) for m in JUDGES}
    for short, col in CRITERIA.items():
        rows.append({"Criterion": short, **{JUDGES[m]: pct(int((df[f"{col} | verdict"] == "YES").sum()), N)
                                            for m, df in per_model.items()}})
    rows.append({"Criterion": "All criteria", **{JUDGES[m]: pct(int((df["overall_verdict"] == "YES").sum()), N)
                                                 for m, df in per_model.items()}})
    combined = pd.read_excel(os.path.join(D, "agentjudge_3model_combined.xlsx"), sheet_name="overall")
    unanimous = combined[(combined.GPT_overall == "YES") & (combined.Claude_overall == "YES") & (combined.Gemini_overall == "YES")]
    rows.append({"Criterion": "Unanimous", JUDGES["GPT"]: pct(len(unanimous), N), JUDGES["Claude"]: "", JUDGES["Gemini"]: ""})
    return pd.DataFrame(rows), unanimous.node_id


def table5(unanimous_ids):
    sim = pd.read_excel(os.path.join(D, "similarity_original_vs_corrected_2024plus.xlsx"))
    top1 = sim.loc[sim.node_id.isin(unanimous_ids), "corrected_max_2024plus"]
    n = len(top1)
    bins = [("≥ 0.90", top1 >= 0.90), ("0.85 – < 0.90", (top1 >= 0.85) & (top1 < 0.90)),
            ("0.80 – < 0.85", (top1 >= 0.80) & (top1 < 0.85)), ("< 0.80", top1 < 0.80)]
    rows, cum = [], 0
    for label, mask in bins:
        k = int(mask.sum()); cum += k
        rows.append({"Similarity range": label, "Count": k, "% of 36": f"{k / n * 100:.1f}%",
                     "Cumulative (≥)": f"{cum} ({cum / n * 100:.1f}%)" if k else "-"})
    print(f"Top-1 similarity of the {n} approved concepts: min {top1.min():.2f}, max {top1.max():.2f}, mean {top1.mean():.2f}")
    return pd.DataFrame(rows)


def save(df, name):
    df.to_csv(os.path.join(T, name + ".csv"), index=False, encoding="utf-8-sig")
    df.to_excel(os.path.join(T, name + ".xlsx"), index=False)
    print(f"\n{name}\n{df.to_string(index=False)}")


if __name__ == "__main__":
    t4, unanimous_ids = table4()
    save(t4, "table4_concept_quality")
    save(table5(unanimous_ids), "table5_similarity_distribution")
