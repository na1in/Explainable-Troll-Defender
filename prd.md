# Product Requirement Document (PRD): Explainable Troll Defender

**Version**: 1.0
**Date**: 2026-01-07
**Target Audience**: Hiring Managers (Senior ML Engineer / AI Researcher roles)
**Hardware Target**: Apple Silicon (M4, 16GB RAM)

---

## 1. Executive Summary
The **Explainable Troll Defender** is an advanced AI system designed not just to detect online toxicity, but to *explain* it. Unlike traditional "black box" classifiers that output a simple probability score (e.g., "98% Toxic"), this system provides **human-readable rationales** citing specific linguistic triggers and context.

This project elevates a standard NLP classification task into a comprehensive **Full-Stack AI Engineering** portfolio piece, demonstrating mastery of **GenAI (SLMs)**, **Explainable AI (XAI)**, and **Production MLOps**.

## 2. Problem Statement
**The "Black Box" Problem**: Standard toxicity classifiers (like the Naive Bayes/SVM models from CS410) suffer from two critical failures:
1.  **Lack of Trust**: Moderators don't know *why* a post was flagged, leading to over-censorship or confusion.
2.  **Nuance Failure**: They struggle with sarcasm, "leetspeak" (e.g., `h@te`), and reclaimed slurs, often misclassifying innocent posts as toxic.

**The Opportunity**: A "Senior" level solution doesn't just predict; it reasons. By providing explanations, we enable human-in-the-loop moderation and build more robust, defensible systems.

## 3. Capabilities & Goals
### 3.1 Core Capabilities
1.  **Detection**: Classify tweets as *Hate Speech*, *Offensive*, or *Normal*.
2.  **Explainability**: Output the specific text span or reasoning that triggered the classification (e.g., "Flagged due to racial slur used in aggressive context").
3.  **Robustness**: Resist simple adversarial attacks (e.g., changing "kill" to "k!ll").

### 3.2 Engineering Goals (The "Resume Wins")
*   **GenAI/LLM Tuning**: Fine-tune a Small Language Model (SLM) like **Phi-3** or **TinyLlama** on local hardware (Mac M4) using quantization.
*   **Data Pipeline**: Process complex nested JSON data (HateXplain) into training tensors.
*   **Production Serving**: Serve the model via a high-performance **FastAPI** endpoint.
*   **Containerization**: Fully Dockerized application for reproducible deployment.
*   **Version Control**: Professional Git workflow (branches per feature) linked to GitHub to demonstrate iterative development.

## 4. Technical Architecture
*   **Hardware**: MacBook M4 (16GB RAM). We will use `MPS` (Metal Performance Shaders) acceleration.
*   **Dataset**: [HateXplain](https://github.com/hate-alert/HateXplain) (20k annotated posts with rationales).
*   **Model Strategy**:
    *   *Primary*: **Generative Approach**. Fine-tune a quantized SLM (Phi-3-mini or similar) to output structured JSON: `{ "label": "hate", "rationale": "..." }`.
    *   *Backup*: **Multi-Head BERT**. One head for classification, one head for token extraction (NER-style).
*   **Ops Stack**: Python 3.11, PyTorch/MLX, FastAPI, Docker.

## 5. Phases of Implementation
This project is structured to verify value at each step.

### Phase 1: Data & Discovery (Days 1-2)
*   **Objective**: Understand the "Why".
*   **Actions**:
    *   Ingest HateXplain dataset.
    *   Visualize "Rationale" distribution (which words trigger toxicity?).
    *   Create train/val/test splits formatted for the chosen model.

### Phase 2: Model Engineering (Days 3-5)
*   **Objective**: Build the "Brain".
*   **Actions**:
    *   Set up local training loop with `MPS` support.
    *   Fine-tune the model (LoRA/QLoRA if needed for memory efficiency on 16GB).
    *   Evaluate using **Ground Truth Rationales** (did the model highlight the right words?).

### Phase 3: MLOps & Serving (Days 6-7)
*   **Objective**: Build the "Product".
*   **Actions**:
    *   Wrap the model in a **FastAPI** service.
    *   Implement **Adversarial Defenses** (input sanitization).
    *   Dockerize the solution.

### Phase 4: Verification & Demo (Day 8)
*   **Objective**: Prove it works.
*   **Actions**:
    *   Run an "Adversarial Attack Suite" script.
    *   Generate a final `walkthrough.md` with screenshots of the model catching tricky examples.
