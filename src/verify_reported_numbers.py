#!/usr/bin/env python3
"""Recompute the headline numbers of the manuscript from the released data files.

Runs offline (no API calls, no GPU) in well under a minute.  Every number is
derived from a file in ``data/`` or ``tables/`` and printed next to the value
quoted in the manuscript, so a reader can confirm the two agree.

    python src/verify_reported_numbers.py

Compressed network files (``*.gz``) are read directly; no unpacking needed.
"""
import gzip
import os
import sys

import networkx as nx
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
D = os.path.join(ROOT, "data")
T = os.path.join(ROOT, "tables")


def p(*parts):
    return os.path.join(D, *parts)


def check(label, value, expected):
    ok = value == expected
    print(f"  {'OK ' if ok else '!! '} {label:<62} {value!s:>14}   (manuscript: {expected})")
    return ok


def read_gexf_any(path):
    if os.path.exists(path):
        return nx.read_gexf(path)
    with gzip.open(path + ".gz", "rb") as f:
        return nx.read_gexf(f)


def read_csv_any(path, **kw):
    return pd.read_csv(path if os.path.exists(path) else path + ".gz", **kw)


def main():
    ok = True
    print("\n[1] Patent corpus")
    seeds = pd.read_csv(p("raw", "collected_data_abstract_with_citation.csv"), dtype=str)
    cited_ids = set()
    for s in seeds["cited_patent_ids"].dropna():
        cited_ids.update(x.strip() for x in s.split(";") if x.strip())
    cited = pd.read_csv(p("raw", "Cited_patent_data.csv"), dtype=str)
    ok &= check("seed patents (haptics x robot, US, filed 2020-2023)", len(seeds), 201)
    ok &= check("unique cited patents referenced by the seeds", len(cited_ids), 5601)
    ok &= check("cited patents with metadata (Cited_patent_data.csv)", len(cited), 5601)
    ok &= check("cited-id sets identical", cited_ids == set(cited["id"]), True)
    ok &= check("subsequent patents in the validation set", len(pd.read_csv(p("raw", "validation_abstract.csv"), usecols=["original_id"])), 120)

    print("\n[2] Co-citation network (undirected, association-strength weights, LCC)")
    before = read_csv_any(p("networks", "cocitation_before_edges.csv"), dtype={"source": str, "target": str})
    after = read_csv_any(p("networks", "cocitation_after_edges.csv"), dtype={"source": str, "target": str})
    ok &= check("nodes before prediction", pd.unique(pd.concat([before.source, before.target])).size, 5090)
    ok &= check("edges before prediction", len(before), 1392858)
    ok &= check("nodes after prediction", pd.unique(pd.concat([after.source, after.target])).size, 5090)
    ok &= check("edges after prediction", len(after), 1392913)
    ok &= check("edges flagged Predicted", int((after.link_status == "Predicted").sum()), 55)

    print("\n[3] Citation network (directed)")
    g_before = read_gexf_any(p("networks", "directed_citation_network_filtered.gexf"))
    g_after = read_gexf_any(p("networks", "network_with_prospective_nodes.gexf"))
    pros = [n for n, a in g_after.nodes(data=True) if a.get("type") == "prospective"]
    ok &= check("nodes / edges before insertion", (g_before.number_of_nodes(), g_before.number_of_edges()), (5210, 6353))
    ok &= check("nodes / edges after insertion", (g_after.number_of_nodes(), g_after.number_of_edges()), (5265, 6463))
    ok &= check("prospective nodes inserted", len(pros), 55)
    ok &= check("every prospective node has degree 2", all(g_after.degree(n) == 2 for n in pros), True)

    print("\n[4] Predicted links and generated concepts")
    links = pd.read_csv(p("predicted_new_links.csv"), dtype={"node1": str, "node2": str})
    concepts = pd.read_csv(p("final_decoded_results.csv"))
    corrected = pd.read_excel(p("prospective_nodes_grammar_corrected.xlsx"))
    ok &= check("predicted links", len(links), 55)
    print(f"      distinct endpoint patents: {pd.unique(pd.concat([links.node1, links.node2])).size}"
          f" | link probability range: {links.probability.min():.3f}-{links.probability.max():.3f}")
    ok &= check("decoded technology concepts (raw vec2text output)", len(concepts), 55)
    ok &= check("grammar-corrected concepts", len(corrected), 55)
    print(f"      concepts whose text changed in grammar correction: {(corrected.decoded_text_original != corrected.decoded_text).sum()}/55")

    print("\n[5] Agent-as-a-judge (GPT / Claude / Gemini; 3 criteria each)")
    judge = pd.read_excel(p("agentjudge_3model_combined.xlsx"), sheet_name="overall")
    unanimous = judge[(judge.GPT_overall == "YES") & (judge.Claude_overall == "YES") & (judge.Gemini_overall == "YES")]
    for m in ("GPT", "Claude", "Gemini"):
        print(f"      {m:<8} overall YES: {(judge[f'{m}_overall'] == 'YES').sum()}/55")
    ok &= check("concepts judged YES on all criteria by all 3 models", len(unanimous), 36)

    print("\n[6] Match with subsequent patents (publication year >= 2024, ada-002 cosine)")
    sim = pd.read_excel(p("similarity_original_vs_corrected_2024plus.xlsx"))
    sim36 = sim[sim.node_id.isin(unanimous.node_id)]
    ok &= check("of the 36, Top-1 cosine >= 0.90", int((sim36.corrected_max_2024plus >= 0.90).sum()), 20)
    ok &= check("of the 36, Top-1 cosine >= 0.85", int((sim36.corrected_max_2024plus >= 0.85).sum()), 33)
    ok &= check("of the 36, Top-1 cosine >= 0.80", int((sim36.corrected_max_2024plus >= 0.80).sum()), 36)
    ok &= check("Appendix B rows (concepts with Top-1 >= 0.90)", len(pd.read_excel(os.path.join(T, "AppendixB_concepts_over_0.90.xlsx"))), 20)
    ok &= check("Table B.1 rows", len(pd.read_excel(os.path.join(T, "TableB1_final.xlsx"), sheet_name="Table B.1")), 20)
    ok &= check("mean shift in Top-1 cosine due to grammar correction",
                round(float((sim.corrected_max_2024plus - sim.orig_max_2024plus).mean()), 4), -0.0001)

    print("\n[7] Technology-area recombination (Table 2)")
    t2 = pd.read_csv(os.path.join(T, "table2_recombination.csv"), encoding="utf-8-sig")
    row = lambda name: int(t2.loc[t2.Recombined_areas == name, "Links_n"].iloc[0])
    ok &= check("cross-area links", row("Subtotal — cross-area"), 46)
    ok &= check("same-area links", row("Subtotal — same-area"), 9)

    print("\n" + ("All checks passed." if ok else "Some checks FAILED - see lines marked !!"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
