# Figures

All PNGs are the files used for the manuscript or its supplementary analyses.
`layouts/` holds the node coordinates (`.npz`) that the network figures reuse, so
the figures can be regenerated without recomputing layouts.

## Manuscript figures

| File | Content | Script |
|---|---|---|
| `figure4_walkthrough.svg` / `.pdf` / `.png` | Figure 4. End-to-end walkthrough of one prospective node (predicted link row 26): (a) the two endpoint patents, (b) their local citation subgraph as collected, two components, (c) the same subgraph after inserting the prospective node, one component, (d) the concept decoded from the node's projected embedding. Vector PDF 179.9 x 69.1 mm; caption in `figure4_caption.tex`. | `src/08_figures/07_make_figure4_svg.py` |
| `fig_walkthrough.png` | Alternative walkthrough: full co-citation network after prediction -> the predicted link A-B (`US-9861271-B2` – `EP-3679851-A1`, p = 0.976) -> the prospective node in the citation network -> the generated concept text. | `05_sub_layout.py`, `06_walkthrough_figure.py` |
| `fig_cocitation_3panel_noverlap.png` | Co-citation network (5,090 nodes, 1.39 M edges) before and after link prediction with an enlarged view of one predicted link; predicted links in red; weak node-overlap removal so the density structure stays visible. | `01`, `02`, `03`, `04_make_network_figures.py` |
| `fig_cocitation_3panel_overlap.png` | Same figure without overlap removal. | same |
| `fig_citation_3panel.png` | Citation network before and after inserting the 55 prospective nodes (red), with enlarged view. | same |

The co-citation network has density 0.108 and mean degree 547; its core looks
dense in any layout. This is a property of the data, not a rendering artefact.

## Evaluation figures

| File | Content | Script |
|---|---|---|
| `specificity_modecollapse.png` | (a) Top-1 cosine to the validation patents for generated concepts vs real patents (leakage-cleaned); (b) how many of the 55 concepts map to each distinct Top-1 target. | `08_plot_specificity.py` |
| `endpoint_sharing_diagnosis.png` | Concept pairs that share an endpoint patent are more similar to each other and more often collapse onto the same Top-1 target. | `09_plot_endpoint_diagnosis.py` |
| `ours_vs_llm_similarity.png` | 55 x 55 cosine heatmap between our structure-only concepts and the content-based LLM descriptions, and the same-link vs cross-link distribution. | `10_plot_ours_vs_llm.py` |
| `head2head_table.png` | Summary table of the head-to-head comparison (resemblance, specificity, diversity, blinded judge). | `11_plot_head2head_table.py` |

## Exploratory figures

| File | Content | Script |
|---|---|---|
| `similarity_distribution.png` | Distribution of cosine similarity between patent embeddings and a domain keyword embedding. | `12_eda_embedding_similarity.py` |
| `predicted_links_similarity_distribution.png`, `predicted_vs_full_similarity_comparison.png` | Similarity distribution of predicted-link endpoints vs the whole corpus. | `13_eda_predicted_links_similarity.py` |
| `tri_group_similarity_comparison.png` | Regular vs predicted-endpoint vs prospective nodes. | `14_eda_prospective_nodes_similarity.py` |
| `centrality_vs_similarity.png` | Degree, betweenness and eigenvector centrality against text-vs-node similarity. | `15_eda_centrality_vs_similarity.py` |

## Regenerating the network figures

```bash
scripts/unpack_data.sh                      # uncompressed edge lists
python src/08_figures/01_layout_citation_igraph.py   # or reuse figures/layouts/*.npz
python src/08_figures/02_layout_cocitation_fr.py
python src/08_figures/03_noverlap.py
python - <<'EOF'
exec(open('src/08_figures/04_make_network_figures.py').read())
build('citation',   'nov_citation.npz')
build('cocitation', 'igfr_cocitation.npz',     suffix='_overlap')
build('cocitation', 'novlight_cocitation.npz', suffix='_noverlap', node_scale=.6)
EOF
```

Layout files: `ig_citation.npz` and `igfr_cocitation.npz` are the raw
Fruchterman-Reingold layouts, `nov_*.npz` after full overlap removal,
`novlight_cocitation.npz` after the weak removal used in the published figure,
`sub_citation.npz` / `sub1_citation.npz` the local subgraph layouts for the
walkthrough.
