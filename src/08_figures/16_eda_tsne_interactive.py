
# Source file in the working repository: EDA/visualize_embeddings_interactive.py
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
from openai import OpenAI
from sklearn.manifold import TSNE
import plotly.express as px
from tqdm import tqdm
import textwrap

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

def format_hover_text(text, width=60, max_chars=1000):
    if pd.isna(text): return "N/A"
    s = str(text)
    if len(s) > max_chars:
        s = s[:max_chars] + "..."
    # Simple HTML escaping
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    return "<br>".join(textwrap.wrap(s, width=width))

def main():
    data_dir = _ROOT + "/data"
    embedding_file = os.path.join(data_dir, _ROOT + "/data/node_embeddings_with_text_embedding_ada.csv")
    tsne_cache_file = os.path.join(data_dir, _ROOT + "/data/tsne_coordinates.csv")
    output_html = os.path.join(data_dir, _ROOT + "/figures/embedding_visualization_detailed.html")
    
    keyword = "Surgical"
    
    print(f"--- Visualizing Embeddings (Target: '{keyword}') ---")
    
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
    df = pd.read_csv(embedding_file)
    # Ensure ID is string
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str)
    
    # Drop duplicates if any
    df.drop_duplicates(subset=['id'], inplace=True)
    print(f"   Loaded {len(df)} rows.")

    # 3. Parse Embeddings & Measure Similarity
    print("3. Parsing embeddings and calculating similarity...")
    matrix = []
    similarities = []
    valid_ids = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        txt_emb_str = row.get('text_embedding')
        row_id = row.get('id')
        try:
            if pd.isna(txt_emb_str):
                continue
            
            if isinstance(txt_emb_str, str):
                vec = ast.literal_eval(txt_emb_str)
            else:
                vec = txt_emb_str
            
            # Check dimensions (Ada-002 is 1536)
            if len(vec) != 1536:
                continue

            matrix.append(vec)
            sim = cosine_similarity(target_emb, vec)
            similarities.append(sim)
            valid_ids.append(row_id)
            
        except Exception as e:
            continue
            
    # Map similarity to ID
    sim_dict = dict(zip(valid_ids, similarities))
    
    # 4. Dimensionality Reduction (t-SNE)
    # Check cache
    run_tsne = True
    tsne_df = None
    
    if os.path.exists(tsne_cache_file):
        print(f"   Cache found: {tsne_cache_file}")
        try:
            tsne_df = pd.read_csv(tsne_cache_file)
            tsne_df['id'] = tsne_df['id'].astype(str)
            
            # Check if cache covers current valid_ids
            cached_ids = set(tsne_df['id'])
            missing_ids = [vid for vid in valid_ids if vid not in cached_ids]
            
            if len(missing_ids) == 0:
                print("   Cache covers all valid IDs. Skipping t-SNE.")
                run_tsne = False
            else:
                print(f"   Cache missing {len(missing_ids)} IDs. Re-running t-SNE.")
        except Exception as e:
            print(f"   Error reading cache: {e}. Re-running t-SNE.")
            
    if run_tsne:
        X = np.array(matrix)
        print(f"   Running t-SNE on {X.shape} matrix... (this may take a few minutes)")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(X)
        
        tsne_df = pd.DataFrame({
            'id': valid_ids,
            'x': X_embedded[:, 0],
            'y': X_embedded[:, 1]
        })
        
        # Save cache
        tsne_df.to_csv(tsne_cache_file, index=False)
        print("   t-SNE complete and cached.")

    # 5. Prepare Plot DataFrame
    # Merge df with tsne_df
    print("5. Preparing visualization data...")
    # Keep only valid IDs from df
    plot_df = df[df['id'].isin(valid_ids)].copy()
    plot_df = plot_df.merge(tsne_df, on='id', how='inner')
    
    # Add Cosine Similarity and Formatted fields
    plot_df['Cosine Similarity'] = plot_df['id'].map(sim_dict)
    
    # Format Title for Header (Wrapped)
    plot_df['Hover_Title'] = plot_df['title'].fillna("Untitled").apply(
        lambda x: "<br>".join(textwrap.wrap(str(x), width=50))
    )
    
    # Format Abstract for Body (Wrapped, Truncated)
    plot_df['Abstract'] = plot_df['abstract'].apply(
        lambda x: format_hover_text(x, width=60, max_chars=1000)
    )

    # 6. Interactive Visualization
    print("6. Generating Plotly chart...")
    
    fig = px.scatter(
        plot_df, 
        x='x', 
        y='y',
        color='Cosine Similarity',
        hover_name='Hover_Title', # Bold header
        hover_data={
            'Hover_Title': False, # Hide distinct column since it's in header
            'Abstract': True,
            'Cosine Similarity': ':.4f',
            'x': False,
            'y': False
        },
        labels={'Hover_Title': 'Title'},
        title=f"2D Projection of Document Embeddings (Colored by Similarity to '{keyword}')",
        color_continuous_scale='RdBu_r',
        height=800
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='t-SNE Dimension 1'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='t-SNE Dimension 2'),
        hoverlabel=dict(align="left") # Align text left
    )
    
    fig.write_html(output_html)
    print(f"   Visualization saved to: {output_html}")

if __name__ == "__main__":
    main()
