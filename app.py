import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
import uvicorn

app = FastAPI(title="Explainable Troll Defender API")

# --- Configuration ---
BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "./phi3-troll-defender/final_adapter"

# Global Model Variables
model = None
tokenizer = None

class SmartFormatter:
    """
    Converts raw JSON evidence into natural language explanations.
    Implements the 'Product Logic' layer.
    """
    @staticmethod
    def format(prediction: dict) -> str:
        label = prediction.get("label", "normal").lower()
        target = prediction.get("target", "None")
        rationale = prediction.get("rationale", "None")
        
        # Scenario 1: Clean Normal
        if label == "normal":
            return "✅ This post is classified as **Normal**. No toxic content detected."
            
        # Scenario 2: Toxic (Hate Speech or Offensive)
        explanation = f"⚠️ Flagged as **{label.title()}**."
        
        # Add Reasoning
        if target and target.lower() != "none":
            explanation += f" It appears to target **{target}**."
        
        if rationale and rationale.lower() != "none":
            # Clean up the rationale (sometimes multiple spans joined by |)
            spans = [s.strip() for s in rationale.split("|")]
            formatted_spans = ", ".join([f"'{s}'" for s in spans])
            explanation += f" The system detected specific toxic language: {formatted_spans}."
            
        # Fallback if no rationale found but labeled toxic
        if explanation == f"⚠️ Flagged as **{label.title()}**.":
            explanation += " The content matches patterns of toxic behavior, though specific triggers were implicit."
            
        return explanation

class TweetRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    target: str
    rationale: str
    explanation: str

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("Loading Model... this might take a minute.")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    
    # Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Load Finetuned Adapter (Try/Except block in case you run this before training finishes)
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print("✅ Successfully loaded Fine-Tuned Adapter.")
    except Exception as e:
        print(f"⚠️ Could not load adapter (Training might still be running). Using Base Model. Error: {e}")
        model = base_model

    model.eval()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TweetRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    # 1. Prompt Engineering (Must match training!)
    user_prompt = f"Analyze the following tweet for toxicity. Provide a JSON response with 'label' (hatespeech, offensive, normal), 'target' (e.g. Women, Muslims), and 'rationale' (specific trigger words). Tweet: {request.text}"
    formatted_prompt = f"<|user|>\n{user_prompt} <|end|>\n<|assistant|>\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # 2. Generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128,
            temperature=0.1, # Low temp for consistent JSON
            do_sample=True
        )
    
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 3. Parsing (Extract the JSON part)
    # The completion includes the user prompt, so we split by <|assistant|> or just look for the last part
    # Simpler approach: find the curly braces
    try:
        # Extract content between { and }
        json_str = completion.split("<|assistant|>")[-1].strip()
        # Find first { and last }
        if "{" in json_str and "}" in json_str:
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            json_clean = json_str[start:end]
            prediction = json.loads(json_clean)
        else:
            raise ValueError("No JSON found")
    except Exception as e:
        # Fallback if model outputs garbage
        print(f"Parsing failed: {completion}")
        prediction = {"label": "error", "target": "unknown", "rationale": "Model output parsing failed"}

    # 4. Smart Formatting
    explanation = SmartFormatter.format(prediction)
    
    return {
        "label": prediction.get("label", "unknown"),
        "target": prediction.get("target", "unknown"),
        "rationale": prediction.get("rationale", "unknown"),
        "explanation": explanation
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
