import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

PROCESSED_DIR = "data/processed"
IMG_DIR = "eda_images"

def run_eda():
    os.makedirs(IMG_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    
    # 1. Label Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title("Distribution of Labels in Training Set")
    plt.savefig(os.path.join(IMG_DIR, "label_distribution.png"))
    print(f"Saved label_distribution.png")
    
    # 2. Text Length
    df['text_len'] = df['text'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='text_len', hue='label', kde=True, bins=30)
    plt.title("Tweet Length (words) by Label")
    plt.savefig(os.path.join(IMG_DIR, "text_length.png"))
    print(f"Saved text_length.png")

    # 3. Rationale Analysis
    hate_df = df[(df['label'].isin(['hate', 'offensive'])) & (df['has_rationale'] == True)]
    all_rationale_text = " ".join(hate_df['rationale'].dropna().astype(str))
    common_words = Counter(all_rationale_text.lower().split()).most_common(20)
    
    words, counts = zip(*common_words)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(words))
    plt.title("Top 20 Trigger Words (Rationales) in Toxic Posts")
    plt.savefig(os.path.join(IMG_DIR, "top_trigger_words.png"))
    print(f"Saved top_trigger_words.png")

    print("EDA Complete. Images saved to eda_images/")

if __name__ == "__main__":
    run_eda()
