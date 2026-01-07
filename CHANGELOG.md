# Changelog

All notable changes to the **Explainable-Troll-Defender** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- **SmartFormatter** logic in `app.py` to convert raw JSON outputs into natural language explanations (e.g., "Flagged as Hate Speech targeting Women...").
- **Hybrid Inference Strategy**: Updated PRD and Plan to reflect the separation of Extraction (Model) and Explanation (Product Layer).
- **Adversarial Test Suite** (`adversarial_test.py`) to verify model robustness against leetspeak and identity attacks.
- **Dockerfile** for production-ready deployment on any container platform.

## [0.2.0] - 2026-01-07
### Added
- **Phase 1 Complete**: Data ingestion pipeline (`data_loader.py`) for HateXplain dataset.
- **EDA Scripts**: `eda.py` generating insights on rationale length and trigger words.
- **Training Script**: `train_generative.py` optimized for Apple Silicon (MPS) using `float16` and `LoRA`.
- **Target Extraction**: Enhanced data loader to parse specific "Target Communities" (e.g., Women, Muslims) from annotator metadata.

## [0.1.0] - 2026-01-07
### Added
- **Project Initialization**: Created repository `Explainable-Troll-Defender`.
- **Documentation**: 
    - `prd.md`: Product Requirement Document defining the "Graduate Level" scope.
    - `implementation_plan.md`: Technical roadmap for LLM + MLOps hybrid approach.
    - `project_upgrade_proposals.md`: Comparison of NLP vs Graph vs MLOps paths.
    - `model_architecture.md`: Conceptual diagram of Generative Instruction Tuning.
- **Version Control**: Linked to GitHub remote `na1in/Explainable-Troll-Defender`.

### Changed
- Selected **HateXplain** as the primary dataset to enable Explainable AI features.
- Selected **Microsoft Phi-3-mini** as the base model for local fine-tuning.
