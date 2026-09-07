
# Source file in the working repository: EDA/analyze_embedding_similarity.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from tqdm import tqdm

# API Key Setting
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def get_embedding(text, model="text-embedding-ada-002"):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None: return 0.0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def main():
    data_dir = _ROOT + "/data"
    embedding_file = os.path.join(data_dir, _ROOT + "/data/node_embeddings_with_text_embedding_ada.csv")
    
    keyword = "surgical, medical, surgery, bone, patient"
    
    print(f"--- Analyzing Similarity with '{keyword}' ---")
    
    # 1. Get Keyword Embedding
    print(f"1. Generating embedding for keyword: '{keyword}'...")
    target_emb = get_embedding(keyword)
    if target_emb is None:
        print("Failed to generate embedding.")
        return

    # 2. Load Data
    if not os.path.exists(embedding_file):
        print(f"File not found: {embedding_file}")
        return
        
    print(f"2. Loading data from {embedding_file}...")
    # Loading only necessary columns to save memory/time if possible
    # We need id, title, abstract (maybe), text_embedding
    try:
        # Use iterator or chunking if really huge, but 300MB fits in memory. 
        # However, parsing 'text_embedding' (string to list) can be slow.
        df = pd.read_csv(embedding_file) 
        print(f"   Loaded {len(df)} rows.")
        
        # Filter out prospective nodes
        initial_rows = len(df)
        if 'id' in df.columns:
            df = df[~df['id'].astype(str).str.contains('prospective node', case=False, na=False)]
            # Also filter out NaN abstracts
            df = df.dropna(subset=['abstract'])
            
            filtered_rows = len(df)
            print(f"   Filtered out rows (prospective node OR missing abstract).")
            print(f"   Remaining rows: {filtered_rows}")
        else:
            print("   'id' column not found, skipping filtering.")
    except Exception as e:
        print(f"   Error loading CSV: {e}")
        return

    # 3. Calculate Similarity
    print("3. Calculating cosine similarity...")
    similarities = []
    
    # Parsing embeddings and calculating similarity
    # tqdm for progress checking
    for txt_emb_str in tqdm(df['text_embedding'], desc="Processing"):
        try:
            if pd.isna(txt_emb_str):
                similarities.append(0.0)
                continue
            
            # Fast parsing attempt
            if isinstance(txt_emb_str, str):
                # Using eval is faster than ast.literal_eval for simple lists, but potential security risk. 
                # Since we trust the file (it's internal data), we can use json or ast. 
                # ast.literal_eval is safer.
                vec = ast.literal_eval(txt_emb_str)
            else:
                vec = txt_emb_str
                
            sim = cosine_similarity(target_emb, vec)
            similarities.append(sim)
        except Exception as e:
            similarities.append(0.0)
            
    df['sim_score'] = similarities
    
    # 4. Statistics and Counts
    print("\n--- Results ---")
    print(f"Statistics for similarity with '{keyword}':")
    print(df['sim_score'].describe())
    
    # Count documents with high similarity
    thresholds = [0.7, 0.75, 0.8, 0.85]
    for th in thresholds:
        count = len(df[df['sim_score'] >= th])
        ratio = count / len(df) * 100
        print(f"Documents with similarity >= {th}: {count} ({ratio:.2f}%)")

    # 5. Plotting
    print("\n4. Generating 2D Similarity Plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sim_score'], bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of Cosine Similarity with '{keyword}'")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.75)
    
    output_plot = _ROOT + "/figures/similarity_distribution.png"
    plt.savefig(output_plot, dpi=300)
    print(f"   Plot saved to: {output_plot}")

    # Top 5 Documents
    print(f"\n[Top 5 Documents Most Similar to '{keyword}']")
    top5 = df.nlargest(5, 'sim_score')
    for i, row in top5.iterrows():
        title = row.get('title', 'No Title')
        score = row['sim_score']
        print(f"[{score:.4f}] {title}")
        
    # Highest and Lowest Documents Details
    print(f"\n[Detailed View of Extremes]")
    
    # Highest
    max_row = df.loc[df['sim_score'].idxmax()]
    print(f"\n>> HIGHEST SIMILARITY DOCUMENT ({max_row['sim_score']:.4f})")
    print(f"Title: {max_row.get('title', 'N/A')}")
    print(f"Abstract: {str(max_row.get('abstract', 'N/A'))[:500]}...") # Truncate abstract if too long

    # Lowest
    min_row = df.loc[df['sim_score'].idxmin()]
    print(f"\n>> LOWEST SIMILARITY DOCUMENT ({min_row['sim_score']:.4f})")
    print(f"Title: {min_row.get('title', 'N/A')}")
    print(f"Abstract: {str(min_row.get('abstract', 'N/A'))[:500]}...")

if __name__ == "__main__":
    main()
