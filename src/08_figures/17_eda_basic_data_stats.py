# Source file in the working repository: EDA/basic_data_stats.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import pandas as pd
import os
import numpy as np
import collections
import re

def get_stopwords():
    return {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at', 
        'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
        'can', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 
        'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 
        'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 
        'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 
        'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 
        'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 
        'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 
        'under', 'until', 'up', 'very', 
        'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 
        'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
    }

def analyze_text_stats(df, column_name):
    if column_name not in df.columns:
        print(f"   [{column_name.upper()}] Column not found.")
        return None, None

    print(f"   [{column_name.upper()}]")
    texts = df[column_name].dropna().astype(str).tolist()
    
    # We will count how many documents contain each word (Document Frequency)
    # But we also want to know total unique words (Vocabulary Size) across corpus
    
    doc_freq_counter = collections.Counter()
    all_unique_words_set = set()
    
    stopwords = get_stopwords()
    
    for text in texts:
        # Simple tokenization: remove non-alphanumeric, split by whitespace, lowercase
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Filter stopwords
        filtered_words = [w for w in words if w not in stopwords]
        
        # Get unique words in THIS document
        unique_words_in_doc = set(filtered_words)
        
        # Update Document Frequency counter
        doc_freq_counter.update(unique_words_in_doc)
        
        # Update global vocabulary set
        all_unique_words_set.update(unique_words_in_doc)
        
    vocab_size = len(all_unique_words_set)
    
    print(f"      Vocabulary Size: {vocab_size}")
    
    if doc_freq_counter:
        most_common, max_count = doc_freq_counter.most_common(1)[0]
        print(f"      [MOST FREQUENT WORD]: '{most_common}' (appears in {max_count} documents)")

    print(f"      Top 20 Frequent Words (by Document Count):")
    top_20 = doc_freq_counter.most_common(20)
    for word, count in top_20:
        print(f"         {word}: {count}")
        
    # Return ALL words for Excel export
    all_words_list = doc_freq_counter.most_common()
    words_df = pd.DataFrame(all_words_list, columns=['Word', 'Frequency'])
    return vocab_size, words_df

def main():
    data_dir = _ROOT + "/data"
    file_path = os.path.join(data_dir, _ROOT + "/data/node_embeddings_with_text_embedding_ada.csv")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Basic Data Statistics for {os.path.basename(file_path)} ---")
    print(f"Loading data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Filter out prospective nodes
    initial_rows = len(df)
    if 'id' in df.columns:
        df = df[~df['id'].astype(str).str.contains('prospective node', case=False, na=False)]
        filtered_rows = len(df)
        print(f"Filtered out {initial_rows - filtered_rows} rows containing 'prospective node' in 'id'.")
    
    summary_data = []

    # 1. Shape
    rows, cols = df.shape
    print(f"\n1. Dataset Shape (Post-Filtering):")
    print(f"   Rows: {rows}")
    print(f"   Columns: {cols}")
    summary_data.append({"Metric": "Rows (Post-Filtering)", "Value": rows})
    summary_data.append({"Metric": "Columns", "Value": cols})

    # 2. Data Types
    print(f"\n2. Column Data Types:")
    print(df.dtypes)
    # Add types to summary? Maybe separate sheet if needed, but simplistic here.

    # 3. Missing Values
    print(f"\n3. Missing Values Count:")
    missing_counts = df.isnull().sum()
    print(missing_counts)
    for col, val in missing_counts.items():
        summary_data.append({"Metric": f"Missing Values ({col})", "Value": val})
    
    # 4. ID Uniqueness
    if 'id' in df.columns:
        num_unique_ids = df['id'].nunique()
        is_unique = num_unique_ids == rows
        print(f"\n4. ID Uniqueness:")
        print(f"   Unique IDs: {num_unique_ids}")
        print(f"   All IDs Unique: {is_unique}")
        summary_data.append({"Metric": "Unique IDs", "Value": num_unique_ids})
        summary_data.append({"Metric": "All IDs Unique", "Value": is_unique})

    # 5. Word Count Statistics for Title and Abstract
    print(f"\n5. Word Count Statistics:")
    word_count_stats = []
    
    for col in ['title', 'abstract']:
        if col in df.columns:
            # Drop NaN for stats calculation
            texts = df[col].dropna().astype(str)
            # Simple whitespace split for word count
            word_counts = texts.apply(lambda x: len(x.split()))
            
            print(f"   [{col.upper()}]")
            if word_counts.empty:
                 print(f"      No data available.")
            else:
                desc = word_counts.describe()
                print(f"      Mean: {desc['mean']:.2f}")
                print(f"      Min:  {desc['min']:.0f}")
                print(f"      25%:  {desc['25%']:.0f}")
                print(f"      50%:  {desc['50%']:.0f} (Median)")
                print(f"      75%:  {desc['75%']:.0f}")
                print(f"      Max:  {desc['max']:.0f}")
                print(f"      Std:  {desc['std']:.2f}")
                
                word_count_stats.append({
                    "Column": col,
                    "Mean": desc['mean'],
                    "Min": desc['min'],
                    "25%": desc['25%'],
                    "Median": desc['50%'],
                    "75%": desc['75%'],
                    "Max": desc['max'],
                    "Std": desc['std']
                })

    # 6. Advanced Text Statistics
    print(f"\n6. Advanced Text Statistics (Vocabulary & Top Words):")
    vocab_title, df_title = analyze_text_stats(df, 'title')
    print("")
    vocab_abstract, df_abstract = analyze_text_stats(df, 'abstract')
    
    # Save to SEPARATE Excel files
    # 1. Title Word Frequency
    if df_title is not None:
        title_excel = _ROOT + "/tables/title_word_frequency.xlsx"
        print(f"\nSaving Title statistics to {title_excel}...")
        df_title.to_excel(title_excel, index=False)
        
    # 2. Abstract Word Frequency
    if df_abstract is not None:
        abstract_excel = _ROOT + "/tables/abstract_word_frequency.xlsx"
        print(f"Saving Abstract statistics to {abstract_excel}...")
        df_abstract.to_excel(abstract_excel, index=False)
        
    # 7. Specific Keyword Coverage Analysis
    print(f"\n7. Specific Keyword Coverage Analysis:")
    target_keywords = ['surgical', 'medical', 'surgery', 'bone', 'patient']
    print(f"   Target Keywords: {target_keywords}")
    
    # Precompute regex for efficiency
    # Pattern: \b(surgical|medical|surgery|bone|patient)\b
    pattern = r'\b(' + '|'.join(target_keywords) + r')\b'
    regex = re.compile(pattern, re.IGNORECASE)
    
    count_covered = 0
    
    # Iterate through rows
    for index, row in df.iterrows():
        title_text = str(row['title']) if pd.notna(row['title']) else ""
        abstract_text = str(row['abstract']) if pd.notna(row['abstract']) else ""
        
        # Check Title
        if regex.search(title_text):
            count_covered += 1
            continue # specific keyword found in title, move to next doc
            
        # Check Abstract
        if regex.search(abstract_text):
            count_covered += 1
            continue
            
    print(f"      Documents containing at least one target keyword: {count_covered}")
    print(f"      Coverage Ratio: {count_covered / len(df) * 100:.2f}%")

    print("Done.")

if __name__ == "__main__":
    main()
