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
| `tables/` | Table 1, Table 2, Table B.1 |
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
python src/verify_reported_numbers.py   # recomputes these from the data files, offline
```

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
