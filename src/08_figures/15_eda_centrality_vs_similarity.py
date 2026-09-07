# Source file in the working repository: EDA/analyze_centrality_vs_similarity.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd
import networkx as nx
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
    network_file = os.path.join(data_dir, _ROOT + "/data/networks/co_citation_network_with_predicted_links.gexf")
    embedding_file = os.path.join(data_dir, _ROOT + "/data/node_embeddings_with_text_embedding_ada.csv")
    
    # 1. Load Co-citation Network
    print(f"1. Loading Network from {network_file}...")
    try:
        G = nx.read_gexf(network_file)
        print(f"   Nodes: {G.number_of_nodes()}")
        print(f"   Edges: {G.number_of_edges()}")
    except Exception as e:
        print(f"   Error loading network: {e}")
        return

    # 2. Calculate Centrality
    print("2. Calculating Centrality Measures...")
    # Calculating on the full graph BEFORE filtering to preserve structural properties
    degree_cent = nx.degree_centrality(G)
    print("   Degree Centrality calculated.")
    
    betweenness_cent = nx.betweenness_centrality(G, k=500, seed=42)
    print("   Betweenness Centrality calculated (Approximated k=500).")
    
    try:
        eigen_cent = nx.eigenvector_centrality(G, max_iter=1000)
        print("   Eigenvector Centrality calculated.")
    except:
        print("   Eigenvector Centrality did not converge (skipping).")
        eigen_cent = {}

    # Convert to DataFrame
    centrality_df = pd.DataFrame({
        'id': list(degree_cent.keys()),
        'Degree': list(degree_cent.values()),
        'Betweenness': [betweenness_cent.get(k, 0) for k in degree_cent.keys()],
        'Eigenvector': [eigen_cent.get(k, 0) for k in degree_cent.keys()]
    })

    # 3. Load Embeddings and Calculate Similarity
    keyword = "surgical, medical, surgery, bone, patient"
    print(f"\n3. Calculating Similarity with '{keyword}'...")
    
    print("   Generating keyword embedding...")
    target_emb = get_embedding(keyword)
    if target_emb is None:
        print("   Failed to get target embedding.")
        return

    print(f"   Loading embeddings from {embedding_file}...")
    try:
        emb_df = pd.read_csv(embedding_file)
        
        # Filtering (Prospective Node & NaN Abstract)
        if 'id' in emb_df.columns:
            emb_df = emb_df[~emb_df['id'].astype(str).str.contains('prospective node', case=False, na=False)]
        emb_df = emb_df.dropna(subset=['abstract'])
        print(f"   Processing {len(emb_df)} valid documents...")

        similarities = []
        ids = []
        
        for idx, row in tqdm(emb_df.iterrows(), total=len(emb_df), desc="Similarity"):
             txt_emb_str = row.get('text_embedding')
             doc_id = row.get('id')
             
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
             ids.append(doc_id)
             
        sim_df = pd.DataFrame({'id': ids, 'Similarity': similarities})
        
    except Exception as e:
        print(f"   Error processing embeddings: {e}")
        return

    # 4. Merge Data
    print("\n4. Merging Data...")
    merged_df = pd.merge(centrality_df, sim_df, on='id', how='inner')
    print(f"   Matched Nodes: {len(merged_df)}")
    
    if len(merged_df) < 5:
        print("   Not enough data points for correlation analysis.")
        return

    # 5. Correlation Analysis
    print("\n5. Correlation Analysis:")
    metrics = ['Degree', 'Betweenness', 'Eigenvector']
    
    for metric in metrics:
        if metric in merged_df.columns and merged_df[metric].sum() > 0:
            pearson_corr, p_val = stats.pearsonr(merged_df[metric], merged_df['Similarity'])
            print(f"   {metric} vs Similarity: Pearson r={pearson_corr:.4f} (p={p_val:.4g})")

    # 6. Visualization
    print("\n6. Generating Plots...")
    plt.figure(figsize=(18, 5))
    
    for i, metric in enumerate(metrics):
        if metric in merged_df.columns:
            plt.subplot(1, 3, i+1)
            sns.regplot(data=merged_df, x=metric, y='Similarity', 
                        scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
            plt.title(f"{metric} Centrality vs Similarity")
            plt.xlabel(f"{metric} Centrality")
            plt.ylabel("Cosine Similarity")
            plt.grid(True, alpha=0.3)
            
    output_plot = _ROOT + "/figures/centrality_vs_similarity.png"
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"   Plots saved to: {output_plot}")

    # 7. High Similarity Group Analysis (>= 0.8)
    print("\n7. High Similarity Group Analysis (Similarity >= 0.8):")
    high_sim_df = merged_df[merged_df['Similarity'] >= 0.8]
    
    if len(high_sim_df) == 0:
        print("   No documents found with Similarity >= 0.8")
    else:
        print(f"   Number of High-Similarity Documents: {len(high_sim_df)}")
        for metric in metrics:
            if metric in merged_df.columns:
                # 1. Mean Centrality of High-Sim Group
                mean_val = high_sim_df[metric].mean()
                
                # 2. Percentile Rank of this Mean in the Overall Distribution
                # We want to know: where does this 'mean_val' fall in the distribution of ALL nodes?
                # Percentile = (Count of nodes with value < mean_val) / Total Nodes * 100
                percentile = stats.percentileofscore(merged_df[metric], mean_val)
                
                print(f"   [{metric}]")
                print(f"      Mean Value (High-Sim Group): {mean_val:.6f}")
                print(f"      Percentile Rank of Mean: {percentile:.2f}% (Top {100-percentile:.2f}%)")

    # Save merged data for inspection
    merged_df.to_csv(_ROOT + "/data/centrality_similarity_data.csv", index=False)
    print(_ROOT + "/data/\n   Data saved to: centrality_similarity_data.csv")

if __name__ == "__main__":
    main()
