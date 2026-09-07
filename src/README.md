# Source code

Each script starts with a header naming the file it came from in the working
repository. Two edits were made for release: API keys are read from environment
variables, and paths are resolved from the repository root via `_ROOT`. Run in
stage order; no arguments.

| Stage | Script | What it does |
|---|---|---|
| 0 | `00_data_acquisition/prepare_seed_and_cited_lists.ipynb` | Merges Google Patents exports and extracts cited-patent ids from the BigQuery export. |
| 1 | `01_network_construction/01_build_cocitation_network.py` | Co-citation network, association-strength weights, largest component. |
| 1 | `01_network_construction/02_build_citation_network.py` | Directed citation network. |
| 2 | `02_link_prediction/01_link_prediction_gcn_xgboost.py` | GCN node embeddings, edge features, 5-model XGBoost ensemble, threshold + per-node cap. Writes `predicted_new_links.csv`. |
| 2 | `02_link_prediction/02_insert_predicted_links_cocitation.py` | Adds predicted links to the co-citation GEXF. |
| 2 | `02_link_prediction/03_insert_prospective_nodes_citation.py` | One prospective node per link in the citation network. |
| 2 | `02_link_prediction/04_threshold_cap_sensitivity.py` | Sensitivity to threshold and cap. |
| 3 | `03_embedding/01_text_embedding_openai_ada.py` | ada-002 embeddings of all abstracts. |
| 3 | `03_embedding/02_text_embedding_gte_local.py` | Local gte-base alternative (robustness run). |
| 3 | `03_embedding/03_graph_embedding_gat.py` | GAT (4 heads, 1,024-d) with frozen text features; learnable features for prospective nodes. Set to the 1,536-d ada configuration of the reported run. |
| 4 | `04_cross_modal_alignment/01_ridge_residual_alignment.py` | Ridge (alpha 1.0) + residual MLP (3 blocks, 512 hidden); loss MSE + 0.5 InfoNCE + 0.5 cosine; seed 42. Set to the ada configuration. |
| 4 | `04_cross_modal_alignment/02_siamese_projection_baseline.py` | Earlier Siamese projector (baseline). |
| 4 | `04_cross_modal_alignment/03_retrain_eval_alignment.py`, `04_projector_reliability.py` | Held-out metrics, multi-seed stability, ridge-only ablation. |
| 5 | `05_concept_generation/01_vec2text_decoding.py` | vec2text ada-002 corrector, 60 steps, beam 4; round-trip check and nearest patent. |
| 5 | `05_concept_generation/02_grammar_correction.py` | Grammar-only correction with `gpt-5.4-mini`. |
| 6 | `06_evaluation/01_subsequent_patent_similarity.py` | Cosine to the 120 validation patents. |
| 6 | `06_evaluation/02_placebo_zscore.py` | Null distributions and permutation test. |
| 6 | `06_evaluation/03_agent_as_a_judge_single_model.py` | Earlier five-criterion judge. |
| 6 | `06_evaluation/04_agent_as_a_judge_multimodel.py`, `05_agent_judge_combine.py` | Three-criterion judge with GPT, Claude, Gemini; merge into the combined table. |
| 6 | `06_evaluation/06_head2head_content_baseline.py`, `07_head2head_multirun.py`, `08_head2head_agentjudge_gpt55.py` | Content-based LLM baseline comparisons. |
| 6 | `06_evaluation/09_ablation_text_dropout.py` | Text-dropout ablation. |
| 6 | `06_evaluation/10_postproc_edit_distance.py`, `11_postproc_gliner_entities.py`, `12_postproc_roundtrip_cosine.py` | Grammar-correction impact checks. |
| 7 | `07_analysis/01_recombination_cpc.py`, `02_table2_recombination.py` | CPC main-group recombination of the 55 links (Table 2). Run 01 then 02. |
| – | `verify_reported_numbers.py` | Recomputes the manuscript's numbers from the data files, offline. |
