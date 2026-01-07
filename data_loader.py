import os
import json
import requests
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

DATASET_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def download_file(url, filepath):
    """Downloads a file from a URL to the specified path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f, tqdm(
        desc=filepath,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def get_majority_label(annotators):
    """Determines the majority label from annotators."""
    labels = [ann['label'] for ann in annotators]
    c = Counter(labels)
    # If there's a tie or no majority, we just take the most common one.
    # In HateXplain, 2/3 agreement is usually required, but we'll take top 1.
    return c.most_common(1)[0][0]

def extract_rationale_text(tokens, rationales):
    """
    Extracts the text corresponding to the rationales.
    Rationales are binary masks over tokens. 
    There can be multiple rationales (from different annotators).
    We will join them with ' | ' to show all perspectives.
    """
    if not rationales:
        return ""
    
    rationale_texts = []
    for rationale_mask in rationales:
        # subset tokens where mask is 1
        # ensure len(rationale_mask) == len(tokens)
        if len(rationale_mask) != len(tokens):
            continue
            
        selected_tokens = [tokens[i] for i in range(len(tokens)) if rationale_mask[i] == 1]
        if selected_tokens:
            rationale_texts.append(" ".join(selected_tokens))
    
    return " | ".join(sorted(list(set(rationale_texts))))

def get_targets(annotators):
    """
    Extracts the target communities mentioned by annotators.
    Returns a sorted, unique list of targets joined by ' | '.
    """
    all_targets = []
    for ann in annotators:
        # 'target' is a list of strings like ['Women', 'African']
        targets = ann.get('target', [])
        all_targets.extend(targets)
    
    if not all_targets:
        return "None"
        
    # Count frequency to pick top targets or just all unique ones?
    # Let's take all unique targets for now.
    unique_targets = sorted(list(set(all_targets)))
    
    # Filter out empty strings or 'None'
    clean_targets = [t for t in unique_targets if t and t.lower() != 'none']
    
    if not clean_targets:
        return "None"
        
    return " | ".join(clean_targets)

def process_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    json_path = os.path.join(RAW_DIR, "dataset.json")
    
    if not os.path.exists(json_path):
        print(f"Downloading dataset to {json_path}...")
        download_file(DATASET_URL, json_path)
    else:
        print(f"Dataset already exists at {json_path}")
        
    print("Parsing JSON...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    records = []
    
    for item_id, item_data in tqdm(data.items(), desc="Processing items"):
        tokens = item_data['post_tokens']
        full_text = " ".join(tokens)
        
        annotators = item_data['annotators']
        
        # Get majority label
        label = get_majority_label(annotators)
        
        # Get targets
        targets = get_targets(annotators)
        
        # Get rationales
        # 'rationales' key exists only if there are rationales
        rationales = item_data.get('rationales', [])
        rationale_text = extract_rationale_text(tokens, rationales)
        
        records.append({
            'id': item_id,
            'text': full_text,
            'label': label,
            'target': targets,
            'rationale': rationale_text,
            'has_rationale': len(rationales) > 0
        })
        
    df = pd.DataFrame(records)
    print(f"Total records: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Save Splits (80/10/10)
    # We shuffle first
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size+val_size]
    test_df = df[train_size+val_size:]
    
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    
    print(f"Saved to {PROCESSED_DIR}: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")

if __name__ == "__main__":
    process_data()
