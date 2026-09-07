# Source file in the working repository: EDA/analyze_predicted_links_similarity.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import ast
from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# API Key Setting (Hardcoded as per existing script)
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
    links_file = os.path.join(data_dir, _ROOT + "/data/predicted_new_links.csv")
    embedding_file = os.path.join(data_dir, _ROOT + "/data/node_embeddings_with_text_embedding_ada.csv")
    
    # 1. Load Predicted Links
    print(f"1. Loading Predicted Links from {links_file}...")
    try:
        links_df = pd.read_csv(links_file)
        # Extract unique nodes involved in predicted links
        unique_nodes = set(links_df['node1'].astype(str)).union(set(links_df['node2'].astype(str)))
        print(f"   Links: {len(links_df)}")
        print(f"   Unique Nodes involved: {len(unique_nodes)}")
    except Exception as e:
        print(f"   Error loading links: {e}")
        return

    # 2. Load Embeddings
    print(f"\n2. Loading Embeddings from {embedding_file}...")
    try:
        emb_df = pd.read_csv(embedding_file)
        
        # Filter 1: Valid Abstracts
        emb_df = emb_df.dropna(subset=['abstract'])
        
        # We will keep ALL nodes for comparison, but mark the ones in predicted links
        if 'id' in emb_df.columns:
            emb_df['is_predicted'] = emb_df['id'].astype(str).isin(unique_nodes)
        else:
            print("   'id' column not found in embeddings.")
            return

        print(f"   Total Nodes with valid data: {len(emb_df)}")
        print(f"   Nodes involved in Predicted Links: {emb_df['is_predicted'].sum()}")
        
    except Exception as e:
        print(f"   Error loading embeddings: {e}")
        return

    # 3. Calculate Similarity
    keyword = "surgical, medical, surgery, bone, patient"
    print(f"\n3. Calculating Similarity with '{keyword}'...")
    
    print("   Generating keyword embedding...")
    target_emb = get_embedding(keyword)
    if target_emb is None:
        print("   Failed to get target embedding.")
        return

    similarities = []
    
    for idx, row in tqdm(emb_df.iterrows(), total=len(emb_df), desc="Similarity"):
         txt_emb_str = row.get('text_embedding')
         
         if pd.isna(txt_emb_str):
             sim = 0.0
         else:
             try:
                 if isinstance(txt_emb_str, str):
                     vec = ast.literal_eval(txt_emb_str)
                 else:
                     vec = txt_emb_str
                 sim = cosine_similarity(target_emb, vec)
             except:
                 sim = 0.0
        
         similarities.append(sim)
         
    emb_df['Similarity'] = similarities

    # 4. Comparative Statistics
    print("\n4. Comparative Statistics:")
    
    # Full Dataset
    print("\n   [FULL DATASET]")
    print(emb_df['Similarity'].describe())
    full_mean = emb_df['Similarity'].mean()
    
    # Predicted Links Subset
    print("\n   [PREDICTED LINKS SUBSET]")
    pred_df = emb_df[emb_df['is_predicted'] == True]
    print(pred_df['Similarity'].describe())
    pred_mean = pred_df['Similarity'].mean()
    
    print(f"\n   Mean Comparison:")
    print(f"      Full Dataset: {full_mean:.4f}")
    print(f"      Predicted Subset: {pred_mean:.4f}")
    print(f"      Difference: {pred_mean - full_mean:.4f}")

    # 5. Comparative Visualization
    print("\n5. Generating Comparative Plot...")
    plt.figure(figsize=(12, 6))
    
    # Plot Full Dataset Distribution
    sns.kdeplot(emb_df['Similarity'], fill=True, color='skyblue', label='Full Dataset (All Nodes)', alpha=0.3)
    
    # Plot Predicted Links Subset Distribution
    sns.kdeplot(pred_df['Similarity'], fill=True, color='red', label='Predicted Links Subset', alpha=0.5)
    
    # Add mean lines
    plt.axvline(full_mean, color='blue', linestyle='dashed', linewidth=1.5, label=f'Full Mean: {full_mean:.2f}')
    plt.axvline(pred_mean, color='darkred', linestyle='dashed', linewidth=1.5, label=f'Predicted Mean: {pred_mean:.2f}')
    
    plt.title(f"Similarity Distribution Comparison\n(Full Dataset vs Predicted Links Subset)")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    output_plot = _ROOT + "/figures/predicted_vs_full_similarity_comparison.png"
    plt.savefig(output_plot, dpi=300)
    print(f"   Plot saved to: {output_plot}")
    
    # Save CSV info
    emb_df[['id', 'Similarity', 'is_predicted']].to_csv(_ROOT + "/data/node_similarity_comparison.csv", index=False)
    print(_ROOT + "/data/   Data saved to: node_similarity_comparison.csv")

if __name__ == "__main__":
    main()
