# Implementation Plan: Explainable Troll Detection System (Hybrid LLM + MLOps)

**Goal**: Build a "Senior Engineer" level portfolio project that detects online toxicity with **explainability** (identifying *why* a post is toxic) and serves it via a **production-grade MLOps pipeline**.

**Dataset**: [HateXplain](https://github.com/hate-alert/HateXplain) (20k items with human rationales).

## User Review Required
> [!IMPORTANT]
> **Compute Requirements**: Fine-tuning an LLM (like Llama-3-8B) requires a GPU. If you are on a Mac M1/M2/M3, we can use `MLX` or `MPS` acceleration. If not, we will fall back to a "Distilled" approach (BERT-based) which runs on any CPU.
> **Scope**: This is a significant expansion. We will create a new directory `Explainable-Troll-Defender` to keep this clean.

## Proposed Architecture

### 1. Data Engineering (The Foundation)
*   **Source**: Download HateXplain raw JSON.
*   **Processing**: Convert nested JSON (Posts -> Annotators -> Rationales) into a training format:
    *   Input: `Tweet Text`
    *   Target 1: `Label` (Hate/Offensive/Normal)
    *   Target 2: `Rationale` (The specific words that triggered the label).

### 2. The Model (Path A: LLM Specialist)
We will implement a **Multi-Task Learning** approach or a **Generative** approach depending on hardware.
*   **Primary Approach (Generative)**: Fine-tune a Small Language Model (SLM) like `Phi-3` or `TinyLlama` to output JSON explanations.
    *   *Input*: "Classify and explain: [Tweet]"
    *   *Output*: `{"class": "Hate", "reasoning": "Uses racial slur '...'"}`
*   **Fallback (Discriminative)**: BERT-based Token Classification (NER style) to highlight toxic phrases.

### 3. MLOps Pipeline (Path C: Engineering)
*   **Serving**: `FastAPI` wrapper around the model.
*   **Containerization**: `Dockerfile` to package the app + dependencies.
*   **Robustness**: Add an "Adversarial Test Suite" (e.g., testing if `h@te` bypasses the filter).

## Proposed Changes

### [New Directory] `Explainable-Troll-Defender/`

#### [NEW] `data_loader.py`
Script to dowload HateXplain and preprocess it into a pandas DataFrame/HuggingFace Dataset.

#### [NEW] `train.py`
Training script using HuggingFace `Trainer` or `PyTorch` loop. Supports LoRA/PEFT if GPU is available.

#### [NEW] `app.py` (FastAPI)
The REST API with endpoints:
*   `POST /predict`: Returns Label + Explanation.
*   `GET /health`: System status.

#### [NEW] `Dockerfile`
Standard Python image setup for deployment.

#### [NEW] `adversarial_test.py`
Script to run "attacks" on the model (character swaps, leetspeak) to prove robustness.

## Verification Plan

### Automated Tests
*   **Data Integrity**: Check if all dataset splits have valid labels/rationales.
*   **API Test**: `pytest` to hit the FastAPI `/predict` endpoint.
*   **Load Test**: Simple `locust` script to measure latency (ms/request).

### Manual Verification
*   We will manually feed "tricky" sarcastic sentences to see if the model catches them vs. the old baseline.
