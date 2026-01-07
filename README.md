# Explainable Troll Defender

**An Advanced, Explainable AI System for Detecting Online Toxicity.**

> [!NOTE]
> This project is a graduate-level evolution of a CS410 coursework project. It moves beyond simple classification to provide **explainable rationales** for why content is flagged, using State-of-the-Art Small Language Models (SLMs) and MLOps best practices.

## Project Vision
Most toxicity classifiers are "black boxes"—they say *what* is toxic, but not *why*. This project solves that by:
1.  **Fine-tuning Generative Models** (Phi-3/Llama-2) to identify specific toxic spans.
2.  Using **Human-Annotated Rationales** from the HateXplain dataset.
3.  Deploying via a robust **FastAPI + Docker** pipeline with adversarial monitoring.

## Architecture
*   **Model**: Fine-tuned Small Language Model (SLM) compatible with Apple Silicon (MPS).
*   **Data**: [HateXplain](https://github.com/hate-alert/HateXplain).
*   **Infrastructure**: FastAPI, Docker, PyTorch/MLX.

## Documentation
*   [Product Requirement Document (PRD)](./prd.md)
*   [Implementation Plan](./implementation_plan.md)
*   [Upgrade Strategy](./project_upgrade_proposals.md)

## Author
*   Nalin
