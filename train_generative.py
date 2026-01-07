import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# Configuration
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "./phi3-troll-defender"
NUM_EPOCHS = 3
BATCH_SIZE = 2 # Small batch for MPS/16GB RAM
LEARNING_RATE = 2e-4

def format_instruction(sample):
    """
    Formats the input into the instruct format Phi-3 expects.
    User: Classify and explain...
    Assistant: JSON...
    """
    tweet = sample['text']
    label = sample['label']
    rationale = sample['rationale'] if pd.notna(sample['rationale']) else "None"
    target = sample['target'] if pd.notna(sample['target']) else "None"
    
    # Prompt Engineering
    user_prompt = f"Analyze the following tweet for toxicity. Provide a JSON response with 'label' (hatespeech, offensive, normal), 'target' (e.g. Women, Muslims), and 'rationale' (specific trigger words). Tweet: {tweet}"
    
    # Target Output
    assistant_response = f'{{"label": "{label}", "target": "{target}", "rationale": "{rationale}"}}'
    
    # Phi-3 Format
    # <|user|>\n{prompt} <|end|>\n<|assistant|>\n{response} <|end|>
    full_text = f"<|user|>\n{user_prompt} <|end|>\n<|assistant|>\n{assistant_response} <|end|>"
    
    return {"text": full_text}


def train():
    print(f"Loading data from data/processed/train.csv...")
    df_train = pd.read_csv("data/processed/train.csv")
    df_val = pd.read_csv("data/processed/val.csv")
    
    # Limit for testing speed (Uncomment to use full data)
    # df_train = df_train.head(500) 
    # df_val = df_val.head(50)

    dataset_train = Dataset.from_pandas(df_train)
    dataset_val = Dataset.from_pandas(df_val)
    
    # Apply formatting
    dataset_train = dataset_train.map(format_instruction)
    dataset_val = dataset_val.map(format_instruction)
    
    print("Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token # Phi-3 specific fix
    
    # Quantization Config (4-bit for memory efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto" # Will find MPS or CPU
    )
    
    # LoRA Config (Parameter Efficient Fine Tuning)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )
    
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True, # MPS supports fp16
        dataset_text_field="text",
        max_seq_length=512,
        packing=False
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    print("Starting Training...")
    trainer.train()
    
    print("Saving Model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))

if __name__ == "__main__":
    train()
