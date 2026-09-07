# Data dictionary

All identifiers are patent publication numbers in Google Patents form
(`US-10383765-B2`, `EP-3679851-A1`). Prospective nodes are named
`prospective node k (row k-1)`, where `row k-1` is the 0-based row of the link in
`predicted_new_links.csv`. Embedding columns hold JSON-style lists of floats.
`SHA256SUMS.txt` lists a checksum for every file in `data/` and `tables/`.

Note on US pre-grant publication numbers: the raw export dropped the leading zero
of the 7-digit serial (`US-2022101745-A1` should read `US-20220101745-A1`). The
data files keep the 10-digit form so that all identifiers stay consistent across
files; `raw/PublicationNumber_corrections.xlsx` lists the 33 numbers printed in the
manuscript with their corrected 11-digit form and Google Patents URL.

## raw/ – inputs

| File | Rows | Columns | Description |
|---|---:|---|---|
| `collected_data_abstract_with_citation.csv` | 201 | original_id, title, abstract, cited_patent_ids, citation_count | Seed patents. Google Patents search `"haptics" AND ("kinesthetic" OR "tactile") AND ("feedback" OR "sensing") AND ("robot")`, US, filing date 2020-01-01 to 2023-12-31, English. Abstracts and backward citations from the Google Patents Public Datasets on BigQuery. `cited_patent_ids` is `;`-separated. |
| `Cited_patent_data.csv` | 5,601 | id, title, abstract, publication_date, cpc_codes | Metadata of every patent cited by at least one seed patent. `cpc_codes` is the full CPC list; `src/07_analysis/01_recombination_cpc.py` derives the representative main group from it. |
| `validation_abstract.csv` | 120 | original_id, title, abstract, cited_patent_ids, citation_count, embedding | Subsequent-patent validation set: patents citing the seed corpus and published after the study period; `embedding` is the ada-002 vector of the abstract. 108 of the 120 have a publication year of 2024 or later and form the reference set for the ">= 2024" similarity. |
| `PublicationNumber_corrections.xlsx` | 33 | As printed, Corrected (11-digit), Google Patents URL | Correction table for the US A1 numbers quoted in the manuscript (see note above). |

## networks/

Files whose name ends in `.gz` are gzipped; run `scripts/unpack_data.sh` to
obtain the uncompressed files the pipeline scripts expect.

| File | Nodes | Edges | Description |
|---|---:|---:|---|
| `patent_co_citation_network_filtered.gexf.gz` | 5,090 | 1,392,858 | Co-citation network of the cited patents, edge weight = association strength, largest connected component. Output of stage 1. |
| `co_citation_network_with_predicted_links.gexf.gz` | 5,090 | 1,392,913 | Same network with the 55 predicted links added; every edge carries `link_status` = `Predicted` (55) or `Unpredicted`. |
| `directed_citation_network_filtered.gexf` | 5,210 | 6,353 | Directed citation network seed -> cited patent. Output of stage 1. |
| `network_with_prospective_nodes.gexf` | 5,265 | 6,463 | Citation network after inserting one prospective node per predicted link. Added nodes have `type = prospective` and degree 2 (the two endpoints of the link); `prospective_value` holds the link probability. Input of the GAT (stage 3). |
| `cocitation_before_edges.csv.gz` | – | 1,392,858 | Edge list of the co-citation network before prediction: source, target, weight. |
| `cocitation_after_edges.csv.gz` | – | 1,392,913 | Edge list after prediction: source, target, weight, link_status. |
| `citation_before_edges.csv` | – | 6,353 | Edge list of the citation network before insertion. |
| `citation_after_edges.csv` | – | 6,463 | Edge list after insertion. |
| `citation_after_nodes.csv` | 5,265 | – | Node table after insertion: id, type (`existing` / `prospective`), prospective_value. |

The edge lists were converted from the GEXF files and have identical node and edge
counts; they load directly in Cosmograph or Gephi.

## Link prediction and embeddings

| File | Rows | Columns | Description |
|---|---:|---|---|
| `predicted_new_links.csv` | 55 | node1, node2, probability | The predicted co-citation links (GCN + XGBoost ensemble, after probability threshold and per-node cap). Row order defines the prospective-node numbering. |
| `prospective_node_projected_embeddings_deep_mlp_ada.csv` | 55 | id, projected_text_embedding | 1,536-d projection of each prospective node's GAT embedding into the ada-002 text space (ridge + residual MLP). Input of vec2text. **This is the run reported in the manuscript.** |
| `prospective_node_projected_embeddings_deep_mlp_gte.csv` | 55 | id, projected_text_embedding | Same, 768-d, for the gte-base robustness run. |
| `prospective_node_projected_embeddings_deep_mlp_con.csv` | 55 | id, projected_text_embedding | Same, 768-d, for the Contriever robustness run. |

## Generated concepts

| File | Rows | Columns | Description |
|---|---:|---|---|
| `final_decoded_results.csv` | 55 | id, projected_text_embedding, decoded_text, round_trip_cosine_sim | Raw vec2text output for each prospective node and the cosine between the target vector and the re-embedded text (round-trip check). |
| `final_decoded_results_with_nn.csv` | 55 | node_id, nn_patent_id, nn_title, nn_abstract, nn_cosine_sim, decoded_text, embedding | Raw output plus the nearest existing patent in the corpus (ada-002 space). `embedding` is the ada-002 vector of the decoded text. |
| `prospective_nodes_grammar_corrected.xlsx` | 55 | node_id, decoded_text_original, decoded_text, len_orig, len_corr, len_ratio | Grammar-corrected concept texts (`gpt-5.4-mini`, temperature 0, no external documents). `decoded_text` is the text shown in the manuscript and judged in the agent-as-a-judge step; 30 of the 55 differ from the raw output. |

## Agent-as-a-judge

| File | Rows | Description |
|---|---:|---|
| `agentjudge_3model_gpt.xlsx` / `_claude.xlsx` / `_gemini.xlsx` | 55 each | Per-model verdict and rationale for each of the three criteria (Novelty & Differentiation, Scientific & Logical Soundness, Significance & Value), `total_yes`, `overall_verdict` (YES only if all three criteria are YES). Models: `gpt-5.4-mini`, `claude-opus-4-8`, `gemini-3.5-flash`. |
| `agentjudge_3model_combined.xlsx` | 55 + 165 | Sheet `overall`: the three overall verdicts side by side, `YES_votes`, `consensus_overall`. The 36 concepts with three YES votes are the "all-agree" set. Sheet `by_criterion`: verdicts per criterion and majority. |
| `AGENTJUDGE_FINAL_REPORT.xlsx` | 6 sheets | Summary counts per criterion and model; the four prompt/text configurations that were compared (raw vs corrected text, with and without the instruction not to penalise surface grammar); the 36 all-agree concepts with their best subsequent-patent match; the similarity distribution; method provenance; rationale for the Gemini prompt condition. |
| `evaluation_results_v8.csv` | 55 | Earlier single-model judge (five criteria, `gpt-4o-mini`) kept for completeness; not used for the reported 36/20 figures. |

## Similarity to subsequent patents

| File | Rows | Description |
|---|---:|---|
| `similarity_original_vs_corrected_2024plus.xlsx` | 55 | For each concept, the maximum ada-002 cosine to the 108 subsequent patents published 2024 or later, computed for the raw text (`orig_max_2024plus`) and the corrected text (`corrected_max_2024plus`), with the best-matching patent, its year and title. Source of the "20 of 36 >= 0.90" figure. |
| `prospective_nodes_with_openai_similarity.xlsx` | 55 | Best-matching validation patent for each concept over all 120 validation patents, with a manual content-match flag (O/X). |
| `prospective_nodes_full_similarity_stats.xlsx` | 55 | Per-concept similarity statistics against all 120 validation patents: mean, std, median, min, max, Top-1 z-score, Top-1..Top-5 titles and scores, counts above 0.9 / 0.8 / 0.7. |
| `prospective_validation_similarity_matrix.xlsx` | 55 x 120 | Full cosine matrix concepts x validation patents (columns are validation-patent titles). |
| `prospective_similarity_summary.xlsx` | 21 | Global summary statistics of the matrix (in Korean labels). |

## Baselines and ablations

| File | Rows | Description |
|---|---:|---|
| `head2head_descriptions.csv` | 55 | Our structure-only concept vs a content-based LLM description generated from both endpoint abstracts, for the same 55 links. |
| `head2head_multirun.csv`, `head2head_multirun_top1.npz` | 55 | Strong content baseline (`gpt-5.1`, 3 seeds) and the Top-1 similarities of ours vs each run, used for the paired bootstrap / Wilcoxon test. |
| `head2head_agentjudge_gpt55.csv` | 2 | Single-judge (`gpt-5.5`) summary for ours vs baseline: mean YES count, coherence, share of HIGH / HIGH+MODERATE verdicts. |
| `ablation_text_dropout.csv`, `ablation_text_dropout.npz` | 55 | Text-dropout ablation: structure-only text vs LLM with abstracts vs LLM with titles only, with Top-1 similarity to the validation set for each. |

## Post-processing checks (raw vs grammar-corrected text)

| File | Rows | Description |
|---|---:|---|
| `postproc_edit_distance.csv` | 55 | Token-level Levenshtein distance between raw and corrected text (substitutions, insertions, deletions, normalised distance, identical flag). |
| `postproc_gliner_entities.csv` | 55 | Technical entities extracted with GLiNER from both versions: kept, new, lost, with the entity lists. |
| `postproc_roundtrip_cosine.csv` | 55 | Cosine between the projected target vector and the re-embedded raw / corrected texts; correction leaves the round-trip cosine essentially unchanged. |

## Exploratory-analysis outputs

| File | Rows | Description |
|---|---:|---|
| `centrality_similarity_data.csv` | 5,748 | Degree, betweenness and eigenvector centrality of each node in the predicted co-citation network with its text-vs-node similarity. |
| `node_similarity_comparison.csv` | 5,897 | Text-vs-node embedding similarity per node, flagged if the node is an endpoint of a predicted link. |
| `tri_group_similarity_comparison.csv` | 6,088 | Same comparison for regular, predicted-endpoint and prospective nodes. |
| `tsne_coordinates.csv` | 5,897 | 2-D t-SNE coordinates of the node embeddings used for the interactive maps. |

## Not included

| File | Size | Why | Regenerate with |
|---|---:|---|---|
| `text_embeddings_ada.csv` | 203 MB | ada-002 vectors of all 5,802 patents; regenerable | `src/03_embedding/01_text_embedding_openai_ada.py` |
| `text_embeddings_gte.csv`, `text_embeddings_con.csv` | 102 MB each | robustness-run counterparts | `src/03_embedding/02_text_embedding_gte_local.py` |
| `node_embeddings_with_text_embedding_ada.csv` (and `_gte`, `_con`) | 210 to 300 MB | GAT node embeddings of all 5,265 nodes | `src/03_embedding/03_graph_embedding_gat.py` |
| `ridge_weights.npz`, `best_residual_correction_model*.pth`, `best_deep_mlp_projection_model.pth` | 3 to 29 MB | trained projector weights | `src/04_cross_modal_alignment/01_ridge_residual_alignment.py` |

These files are available from the authors on request.
