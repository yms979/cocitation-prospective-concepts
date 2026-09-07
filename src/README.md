# Code

Each file names its source in the working repository in its header. Release edits:
API keys come from environment variables; paths resolve from the repository root.

| Manuscript section | Script |
|---|---|
| 4.1 Data acquisition | `00_data_acquisition/prepare_seed_and_cited_lists.ipynb` |
| 4.2 Co-citation network | `01_network_construction/01_build_cocitation_network.py` |
| 4.2 Link prediction (GNN embeddings + 5 XGBoost, threshold 0.5, cap 3) | `02_link_prediction/01_link_prediction_gcn_xgboost.py` |
| 4.2 Expanded co-citation network | `02_link_prediction/02_insert_predicted_links_cocitation.py` |
| 4.3 Citation network | `01_network_construction/02_build_citation_network.py` |
| 4.3 Prospective node insertion | `02_link_prediction/03_insert_prospective_nodes_citation.py` |
| 4.4 Text embedding (ada-002) | `03_embedding/01_text_embedding_openai_ada.py` |
| 4.3 GAT node embedding (1,024-d) | `03_embedding/02_graph_embedding_gat.py` |
| 4.4 Cross-modal alignment (ridge + residual MLP) | `04_cross_modal_alignment/01_ridge_residual_alignment.py` |
| 4.4 Embedding inversion (vec2text) | `05_concept_generation/01_vec2text_decoding.py` |
| 4.4 Post-processing (gpt-5.4-mini) | `05_concept_generation/02_grammar_correction.py` |
| 4.5 Table 2 | `07_analysis/01_recombination_cpc.py`, then `02_table2_recombination.py` |
| 5.2 Concept quality assessment | `06_evaluation/01_agent_as_a_judge_multimodel.py`, then `02_agent_judge_combine.py` |
| 5.3 Future alignment assessment | `06_evaluation/03_subsequent_patent_similarity.py` |
| Check reported numbers | `verify_reported_numbers.py` |
