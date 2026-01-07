import requests
import json
import pytest

# Target URL (ensure app is running)
URL = "http://localhost:8000/predict"

def get_prediction(text):
    try:
        response = requests.post(URL, json={"text": text})
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# 1. Basic Toxicity Test
def test_clear_toxicity():
    text = "You are a stupid bitch and deserve to die."
    print(f"Testing: {text}")
    res = get_prediction(text)
    print(f"Result: {json.dumps(res, indent=2)}")
    # We expect 'offensive' or 'hatespeech'
    assert res.get('label') in ['hatespeech', 'offensive']

# 2. Adversarial Attack (Leetspeak)
def test_leetspeak_attack():
    text = "You are a b1tch and I will k1ll you."
    print(f"Testing: {text}")
    res = get_prediction(text)
    print(f"Result: {json.dumps(res, indent=2)}")
    # A good model should still catch this
    # Note: Base Phi-3 might struggle without fine-tuning on this, 
    # but our HateXplain training might help if it had noisy data.

# 3. False Positive Check (Counter-statement)
def test_counter_speech():
    text = "It is wrong to call someone a bitch."
    print(f"Testing: {text}")
    res = get_prediction(text)
    print(f"Result: {json.dumps(res, indent=2)}")
    # Should ideally be 'normal' because it mentions the slur but uses it clinically.

# 4. Identity Attack
def test_identity_hate():
    text = "Go back to your country you invader."
    print(f"Testing: {text}")
    res = get_prediction(text)
    
    # Check if Smart Formatter extracted the target
    if 'target' in res and res['target'] != 'unknown':
        print(f"✅ Correctly identified target: {res['target']}")
    else:
        print("⚠️ Failed to identify target.")

if __name__ == "__main__":
    print("Running Manual Adversarial Check...")
    test_clear_toxicity()
    print("-" * 20)
    test_leetspeak_attack()
    print("-" * 20)
    test_counter_speech()
    print("-" * 20)
    test_identity_hate()
