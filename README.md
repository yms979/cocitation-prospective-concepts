# Decoding predicted links in patent co-citation networks

Data and code for *Decoding predicted links in patent co-citation networks:
Cross-modal generation of prospective technology concepts*.

![Framework](docs/framework.jpg)

## Contents

| Path | What it holds |
|---|---|
| `data/raw/` | 201 seed patents, 5,601 cited patents, validation patents |
| `data/networks/` | co-citation and citation networks before and after prediction |
| `data/` | 55 predicted links, projected embeddings, generated concepts, judge verdicts, similarity to subsequent patents |
| `tables/` | Table 1, Table 2, Table 4, Table 5, Table B.1 |
| `src/` | code for Stage 1 to 3, the concept quality assessment, the future alignment assessment, and Table 2 |

## Reported numbers

| | Value |
|---|---:|
| Seed patents / cited patents | 201 / 5,601 |
| Co-citation network | 5,090 nodes, 1,392,858 edges |
| Citation network before / after prospective nodes | 5,210 / 5,265 nodes |
| Predicted links = generated concepts | 55 |
| Concepts passing all three judges | 36 |
| Concepts with Top-1 cosine >= 0.90 to a subsequent patent | 20 |

```bash
python src/verify_reported_numbers.py            # recomputes these from the data files, offline
python src/06_evaluation/04_validation_tables.py # rebuilds Table 4 and Table 5 in tables/
```

## Validation results

Table 4. Concept quality assessment (n = 55; passes per criterion).

| Criterion | gpt-5.4-mini | claude-opus-4.8 | gemini-3.5-flash |
|---|---:|---:|---:|
| Novelty | 47 (85.5%) | 50 (90.9%) | 54 (98.2%) |
| Plausibility | 55 (100%) | 53 (96.4%) | 50 (90.9%) |
| Significance | 52 (94.5%) | 52 (94.5%) | 44 (80.0%) |
| All criteria | 46 (83.6%) | 47 (85.5%) | 42 (76.4%) |
| Unanimous | 36 (65.5%) | | |

Table 5. Top-1 cosine similarity between the 36 approved concepts and their most
similar subsequent patents.

| Similarity range | Count | % of 36 | Cumulative (≥) |
|---|---:|---:|---:|
| ≥ 0.90 | 20 | 55.6% | 20 (55.6%) |
| 0.85 – < 0.90 | 13 | 36.1% | 33 (91.7%) |
| 0.80 – < 0.85 | 3 | 8.3% | 36 (100%) |
| < 0.80 | 0 | 0% | – |

Per-concept verdicts and rationales are in `data/agentjudge_3model_*.xlsx`; the
matched patents for the 20 concepts at or above 0.90 are in `tables/TableB1_final.xlsx`.

## Running

```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.7.0 && pip install -r requirements.txt
cp .env.example .env            # OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
set -a; source .env; set +a
scripts/unpack_data.sh          # decompress the network files
```

Run the scripts in `src/` in numbered order; they take no arguments. Stages 3 to 5
need a GPU and an OpenAI key; the judge needs OpenAI, Anthropic and Google keys.
Full text-embedding and node-embedding matrices and trained projector weights are
not included; stages 3 and 4 regenerate them.

Code MIT, data CC BY 4.0. Patent metadata from Google Patents Public Data (BigQuery).
