# Source file in the working repository: EDA/analyze_predicted_links_text_stats.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd
import os
import re
import sys

# Add the directory to sys.path to import basic_data_stats
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from basic_data_stats import analyze_text_stats, get_stopwords

def main():
    data_dir = _ROOT + "/data"
    links_file = os.path.join(data_dir, _ROOT + "/data/predicted_new_links.csv")
    text_file = os.path.join(data_dir, _ROOT + "/data/node_embeddings_with_text_embedding_ada.csv")
    
    print(f"1. Loading Predicted Links from {links_file}...")
    try:
        links_df = pd.read_csv(links_file)
        predicted_nodes = set(links_df['node1'].astype(str)).union(set(links_df['node2'].astype(str)))
        print(f"   Unique Predicted Nodes: {len(predicted_nodes)}")
    except Exception as e:
        print(f"   Error loading links: {e}")
        return

    print(f"\n2. Loading Text Data from {text_file}...")
    try:
        df = pd.read_csv(text_file)
        # Filter for predicted nodes
        if 'id' in df.columns:
            df = df[df['id'].astype(str).isin(predicted_nodes)]
        else:
            print("   'id' column not found.")
            return
            
        print(f"   Nodes Found with Text Data: {len(df)}")
    except Exception as e:
         print(f"   Error loading text data: {e}")
         return

    # Analyze Title
    print("\n3. Title Analysis:")
    vocab_title, df_title = analyze_text_stats(df, 'title')
    
    # Analyze Abstract
    print("\n4. Abstract Analysis:")
    vocab_abstract, df_abstract = analyze_text_stats(df, 'abstract')
    
    # Save Excel
    if df_title is not None:
        title_excel = _ROOT + "/tables/predicted_links_title_word_frequency.xlsx"
        print(f"\nSaving Title statistics to {title_excel}...")
        df_title.to_excel(title_excel, index=False)

    if df_abstract is not None:
        abstract_excel = _ROOT + "/tables/predicted_links_abstract_word_frequency.xlsx"
        print(f"Saving Abstract statistics to {abstract_excel}...")
        df_abstract.to_excel(abstract_excel, index=False)

    # Keyword Coverage
    print(f"\n5. Specific Keyword Coverage Analysis:")
    target_keywords = ['surgical', 'medical', 'surgery', 'bone', 'patient']
    print(f"   Target Keywords: {target_keywords}")
    
    pattern = r'\b(' + '|'.join(target_keywords) + r')\b'
    regex = re.compile(pattern, re.IGNORECASE)
    
    count_covered = 0
    for index, row in df.iterrows():
        title_text = str(row['title']) if pd.notna(row['title']) else ""
        abstract_text = str(row['abstract']) if pd.notna(row['abstract']) else ""
        
        if regex.search(title_text) or regex.search(abstract_text):
            count_covered += 1
            
    print(f"      Documents containing at least one target keyword: {count_covered}")
    print(f"      Coverage Ratio: {count_covered / len(df) * 100:.2f}%")

if __name__ == "__main__":
    main()
