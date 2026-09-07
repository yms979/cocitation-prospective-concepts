# Source code

Scripts are numbered by pipeline stage and, within a stage, by execution order.
Each file starts with a header naming the script it was derived from in the
working repository and the two edits made for release: API keys are read from
environment variables, and file paths are resolved relative to the repository root
through a `_ROOT` constant. Apart from that, and the configuration notes below,
the code is the code that produced the shipped results.

Run every script from the repository root (or anywhere; paths are absolute at
runtime). Load the API keys first: `set -a; source .env; set +a`.

## 00_data_acquisition

| File | Purpose |
|---|---|
| `prepare_seed_and_cited_lists.ipynb` | Merges the Google Patents search exports, extracts the unique cited-patent identifiers from the BigQuery export, and builds the seed and cited-patent tables. The Google Patents query and the BigQuery source are documented in `data/README.md`. |

## 01_network_construction

| File | Purpose |
|---|---|
| `01_build_cocitation_network.py` | Co-citation network from the seed patents' citation lists; edge weight = association strength; keeps the largest connected component. |
| `02_build_citation_network.py` | Directed citation network seed -> cited patent. |

## 02_link_prediction

| File | Purpose |
|---|---|
| `01_link_prediction_gcn_xgboost.py` | Core of stage 2. Trains a 2-layer GCN (128 / 64) on the co-citation graph, builds edge features for unconnected pairs, scores them with an ensemble of five XGBoost models, and keeps links above the probability threshold subject to a per-node cap. Writes `data/predicted_new_links.csv`. |
| `02_insert_predicted_links_cocitation.py` | Adds the predicted links to the co-citation GEXF with `link_status`. |
| `03_insert_prospective_nodes_citation.py` | Replaces each predicted link by one prospective node connected to both endpoints in the citation network. |
| `04_threshold_cap_sensitivity.py` | Sensitivity of the number of predicted links and endpoints to the probability threshold and the per-node cap (manuscript Section 4.7). |

## 03_embedding

| File | Purpose |
|---|---|
| `01_text_embedding_openai_ada.py` | ada-002 embedding of every seed and cited abstract, with exponential back-off. Output is not resumable; an existing output file is not overwritten but numbered. |
| `02_text_embedding_gte_local.py` | Local `thenlper/gte-base` alternative (768-d) used for the robustness run. |
| `03_graph_embedding_gat.py` | Core of stage 3. GAT (4 heads, 1,024-d output) on the citation network with prospective nodes; nodes with an abstract start from their frozen text embedding through a 1,536 -> 256 projection, prospective nodes from a learnable vector. Early stopping, patience 20. **Configuration note:** the release copy sets `TEXT_EMBEDDING_DIM = 1536` and the `_ada` file names (the reported run); the last working-copy edit had the 768-d gte setting. |

## 04_cross_modal_alignment

| File | Purpose |
|---|---|
| `01_ridge_residual_alignment.py` | Core of stage 4. Stage A ridge regression (closed form, alpha 1.0) from node to text space; stage B residual MLP (3 blocks, 512 hidden, dropout 0.1) trained on the residual with loss MSE + 0.5 InfoNCE (tau 0.07) + 0.5 cosine; 80/20 split with `random_state=42`, patience 50. Writes the projected embeddings of the prospective nodes. **Configuration note:** as above, the release copy carries the ada-002 (1,536-d) setting. |
| `02_siamese_projection_baseline.py` | Earlier Siamese-network projector, kept as the baseline it was compared against. |
| `03_retrain_eval_alignment.py` | Re-trains the projector and reports scale-invariant metrics on the held-out split (per-dimension MSE, cosine, self-retrieval Top-1/5/10 and median rank vs a random baseline). |
| `04_projector_reliability.py` | Multi-seed stability of the projector and ridge-only vs ridge + residual ablation, measured on projected embeddings before decoding. |

## 05_concept_generation

| File | Purpose |
|---|---|
| `01_vec2text_decoding.py` | Core of stage 5. Inverts each projected vector with the vec2text ada-002 corrector (60 steps, sequence beam width 4, batch size 1), re-embeds the text for a round-trip check, finds the nearest real patent, and prints quality statistics. vec2text calls the OpenAI embedding API internally at each correction step. |
| `02_grammar_correction.py` | Grammar and typo correction of the 55 concepts with `gpt-5.4-mini` at temperature 0. The prompt forbids adding, removing or inventing content and any reference to external documents. |

## 06_evaluation

| File | Purpose |
|---|---|
| `01_subsequent_patent_similarity.py` | Cosine similarity between each concept and the 120 validation patents; writes the per-concept statistics, the full matrix and the summary. |
| `02_placebo_zscore.py` | Null distributions and permutation test: do the decoded concepts match the validation frontier better than random existing patents and than random node pairs, with leakage control. |
| `03_agent_as_a_judge_single_model.py` | First agent-as-a-judge (five criteria, LangGraph, Semantic Scholar search); output `evaluation_results_v8.csv`. |
| `04_agent_as_a_judge_multimodel.py` | The judge used for the reported results: three criteria, YES/NO with rationale, one reflection round with Semantic Scholar search limited to 2023 and earlier, run with GPT, Claude and Gemini. Reads the grammar-corrected concepts; supports resume via `JUDGE_RESUME=1`. |
| `05_agent_judge_combine.py` | Merges the three per-model files into `agentjudge_3model_combined.xlsx`, marking unavailable models and computing consensus from the models that answered. |
| `06_head2head_content_baseline.py` | Content-based LLM baseline (given both endpoint abstracts) vs ours on identical metrics and a blinded judge. |
| `07_head2head_multirun.py` | Stronger baseline with `gpt-5.1`, three seeds, paired bootstrap and Wilcoxon test on the per-link Top-1 difference. |
| `08_head2head_agentjudge_gpt55.py` | Runs the single-model judge logic with `gpt-5.5` on both ours and the baseline for a consistent evaluator. |
| `09_ablation_text_dropout.py` | Text-dropout ablation: structure-only vs LLM with abstracts vs LLM with titles only, paired Wilcoxon. |
| `10_postproc_edit_distance.py` | Token-level edit distance raw vs corrected concept text. |
| `11_postproc_gliner_entities.py` | Technical-entity retention with GLiNER (`urchade/gliner_medium-v2.1`). |
| `12_postproc_roundtrip_cosine.py` | Round-trip cosine of raw vs corrected text against the projected target vector. |

## 07_analysis

| File | Purpose |
|---|---|
| `01_recombination_cpc.py` | Representative CPC main group per endpoint patent (first non-indexing, non-Y code), the recombined area pair of every predicted link, and the baseline share of each pair among the existing co-citation links. Writes an intermediate pickle to `data/`. |
| `02_table2_recombination.py` | Builds Table 2 from the pickle (cross-area and same-area blocks, shares vs existing links) and writes `tables/table2_recombination.{xlsx,csv}`. |

Run `01` then `02`; the result is byte-identical to the shipped table.

## 08_figures

| File | Purpose |
|---|---|
| `01_layout_citation_igraph.py`, `02_layout_cocitation_fr.py` | Fruchterman-Reingold layouts (igraph) for the citation and co-citation networks, computed on the *after* graph so that before/after panels share coordinates. Saved to `figures/layouts/`. |
| `03_noverlap.py` | Node-overlap removal (weak setting for the dense co-citation network, so that density gradients survive). |
| `04_make_network_figures.py` | Three-panel before / after / enlarged-view figures for both networks. |
| `05_sub_layout.py`, `06_walkthrough_figure.py` | Local subgraph around one predicted link and the walkthrough figure from network to decoded text. |
| `07_make_figure4_svg.py` | Figure 4 (SVG with PDF and PNG exports): endpoint patents, local citation subgraph before and after insertion, decoded concept. |
| `08_plot_specificity.py` | Specificity of generated vs real patents and Top-1 target diversity. |
| `09_plot_endpoint_diagnosis.py` | Does endpoint sharing explain the partial mode collapse. |
| `10_plot_ours_vs_llm.py`, `11_plot_head2head_table.py` | Structure-only vs content-baseline comparison figure and summary table. |
| `12_..._18_eda_*.py` | Exploratory analyses: similarity distributions, predicted-link subsets, centrality vs similarity, t-SNE map, word statistics. Several of these need the full node-embedding matrix (not shipped) and an OpenAI key. |

## verify_reported_numbers.py

Recomputes every headline number of the manuscript from the shipped files and
prints them next to the quoted values. No API, no GPU, about one minute.

## Excluded from the release

Pilot-study iterations on other domains, superseded scripts (`_보류_*`, clique-based
variants, the OpenAlex downloaders from an earlier paper-based pilot, an earlier
LangGraph validation script), backups and logs. None of them contributed to the
reported results.
