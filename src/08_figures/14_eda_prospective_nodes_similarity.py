# Source file in the working repository: EDA/analyze_prospective_nodes_similarity.py
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
from scipy import stats

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
    embedding_file = os.path.join(data_dir, _ROOT + "/data/node_embeddings_with_text_embedding_ada.csv")
    prospective_file = os.path.join(data_dir, _ROOT + "/data/prospective_node_projected_embeddings_deep_mlp_ada.csv")
    links_file = os.path.join(data_dir, _ROOT + "/data/predicted_new_links.csv")
    
    # 0. Load Predicted Links Nodes
    print(f"0. Loading Predicted Links nodes from {links_file}...")
    predicted_nodes = set()
    try:
        links_df = pd.read_csv(links_file)
        predicted_nodes = set(links_df['node1'].astype(str)).union(set(links_df['node2'].astype(str)))
        print(f"   Unique Predicted Nodes: {len(predicted_nodes)}")
    except Exception as e:
        print(f"   Error loading links: {e}")

    # 1. Load Regular Embeddings
    print(f"1. Loading Regular Embeddings from {embedding_file}...")
    try:
        reg_df = pd.read_csv(embedding_file)
        # Filter for Valid Abstracts & Non-Prospective
        reg_df = reg_df.dropna(subset=['abstract'])
        if 'id' in reg_df.columns:
            reg_df = reg_df[~reg_df['id'].astype(str).str.contains('prospective node', case=False, na=False)]
            # Flag Predicted Nodes
            reg_df['is_predicted'] = reg_df['id'].astype(str).isin(predicted_nodes)
        else:
            print("   'id' column not found.")
            return
        
        reg_df['is_prospective'] = False
        print(f"   Regular Nodes: {len(reg_df)}")
        print(f"   (Included Predicted Nodes: {reg_df['is_predicted'].sum()})")

    except Exception as e:
        print(f"   Error loading regular embeddings: {e}")
        return

    # 2. Load Prospective Embeddings
    print(f"2. Loading Prospective Embeddings from {prospective_file}...")
    try:
        pros_df = pd.read_csv(prospective_file)
        # Rename column to match
        pros_df = pros_df.rename(columns={'projected_text_embedding': 'text_embedding'})
        pros_df['is_prospective'] = True
        pros_df['is_predicted'] = False # Prospective nodes are new, not in the old predicted links file usually
        
        # We don't have abstracts for these, but we have embeddings, which is enough for similarity.
        print(f"   Prospective Nodes: {len(pros_df)}")
        
    except Exception as e:
        print(f"   Error loading prospective embeddings: {e}")
        return

    # Combine
    emb_df = pd.concat([reg_df, pros_df], ignore_index=True)
    embeddings_count = len(emb_df)

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
    
    # Regular Nodes
    print("\n   [REGULAR NODES]")
    reg_df = emb_df[emb_df['is_prospective'] == False]
    print(reg_df['Similarity'].describe())
    reg_mean = reg_df['Similarity'].mean()
    
    # Predicted Link Nodes (Subset of Regular)
    print("\n   [PREDICTED LINKS SUBSET]")
    pred_df = reg_df[reg_df['is_predicted'] == True]
    print(pred_df['Similarity'].describe())
    pred_mean = pred_df['Similarity'].mean()

    # Prospective Nodes
    print("\n   [PROSPECTIVE NODES]")
    pros_df = emb_df[emb_df['is_prospective'] == True]
    print(pros_df['Similarity'].describe())
    pros_mean = pros_df['Similarity'].mean()
    
    print(f"\n   Mean Comparison:")
    print(f"      Regular Mean: {reg_mean:.4f}")
    print(f"      Predicted Subset Mean: {pred_mean:.4f}")
    print(f"      Prospective Mean: {pros_mean:.4f}")
    print(f"      Diff (Pros - Reg): {pros_mean - reg_mean:.4f}")
    print(f"      Diff (Pred - Reg): {pred_mean - reg_mean:.4f}")
    
    # T-test (Prospective vs Regular)
    t_stat, p_val = stats.ttest_ind(reg_df['Similarity'], pros_df['Similarity'], equal_var=False)
    print(f"   T-test (Pros vs Reg): t={t_stat:.4f}, p={p_val:.4g}")

    # 5. Visualization
    print("\n5. Generating Comparative Plot...")
    plt.figure(figsize=(12, 6))
    
    # Plot Regular Nodes Distribution
    sns.kdeplot(reg_df['Similarity'], fill=True, color='skyblue', label=f'Regular Nodes (All) (n={len(reg_df)})', alpha=0.3)
    
    # Plot Predicted Links Subset
    sns.kdeplot(pred_df['Similarity'], fill=True, color='red', label=f'Predicted Links Subset (n={len(pred_df)})', alpha=0.4)

    # Plot Prospective Nodes Distribution
    sns.kdeplot(pros_df['Similarity'], fill=True, color='orange', label=f'Prospective Nodes (n={len(pros_df)})', alpha=0.5)
    
    # Add mean lines
    plt.axvline(reg_mean, color='blue', linestyle='dashed', linewidth=1.5, label=f'Reg Mean: {reg_mean:.2f}')
    plt.axvline(pred_mean, color='darkred', linestyle='dashed', linewidth=1.5, label=f'Pred Mean: {pred_mean:.2f}')
    plt.axvline(pros_mean, color='darkorange', linestyle='dashed', linewidth=1.5, label=f'Pros Mean: {pros_mean:.2f}')
    
    plt.title(f"Similarity Comparison: Regular vs Predicted vs Prospective")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    output_plot = _ROOT + "/figures/tri_group_similarity_comparison.png"
    plt.savefig(output_plot, dpi=300)
    print(f"   Plot saved to: {output_plot}")
    
    # Save CSV info
    emb_df[['id', 'Similarity', 'is_prospective', 'is_predicted']].to_csv(_ROOT + "/data/tri_group_similarity_comparison.csv", index=False)
    print(_ROOT + "/data/   Data saved to: tri_group_similarity_comparison.csv")

if __name__ == "__main__":
    main()
