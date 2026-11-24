# 🎯 HireSync: Deep Learning Based Skill Mapping & Resume-to-Job Matching System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-yellow.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/HuggingFace-resume--job--description--fit-orange.svg)](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)

> **Course:** Deep Learning (Sem-07)  
> **Topic:** 12 — Resume-Job Description Matching System (**SIH26044**)  
> **Author:** Praveen Kumar Yadav (Reg: 12304556, Roll: 7)

---

## 📌 1. Project Overview & Objective

In modern recruitment systems, traditional keyword-matching Applicant Tracking Systems (ATS) suffer from the **Vocabulary Mismatch Problem** — qualified candidates are routinely rejected simply because their resume uses different phrasing or synonyms (e.g., writing *"34 automated async tests with pytest"* instead of the literal keyword *"testing"*).

**HireSync** solves this by implementing an end-to-end **Deep Learning Semantic Matching and Skill Gap Analysis System**. It replaces naive keyword filtering with a **Two-Tier Architecture**:
1. **Sentence-BERT (`all-MiniLM-L6-v2`) Dense Embeddings**: Captures rich semantic and contextual intent of full resumes and job descriptions.
2. **Learned PyTorch Neural Re-Ranking Layer (MLP)**: Takes high-dimensional cross-interaction features and explicit skill overlap to predict a calibrated **0–100% match score**.
3. **Semantic Skill Gap Extractor**: Discovers which required job skills are present (**Emerald Green**) and which are missing (**Crimson Red**), using global vector-similarity soft matching.

---

## 🏗️ 2. System Architecture

```
                    ┌─────────────────────────┐
                    │ Candidate Resume (PDF)  │
                    └────────────┬────────────┘
                                 │ Text Extraction (pypdf)
                                 ▼
                    ┌─────────────────────────┐
                    │ Clean Resume Text       │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
       ┌───────────────────┐           ┌────────────────────┐
       │ SBERT Embedding   │           │ SBERT Embedding    │
       │ Vector u ∈ R³⁸⁴   │           │ Vector v ∈ R³⁸⁴    │
       └─────────┬─────────┘           └─────────┬──────────┘
                 │                               │
                 │   Job Description (Cleaned)   │
                 └───────────────┬───────────────┘
                                 ▼
       ┌────────────────────────────────────────────────────┐
       │ Multi-Domain Taxonomy & SBERT Semantic Soft-Match  │
       │  - Matched Skills (Green)                          │
       │  - Missing Skills (Red Gap)                        │
       │  - Overlap Ratio: |S_res ∩ S_jd| / |S_jd|          │
       └─────────────────────────┬──────────────────────────┘
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────┐
       │ Cross-Interaction Feature Vector (1537 dimensions) │
       │ x = [ u,  v,  |u - v|,  u ⊙ v,  overlap_ratio ]    │
       └─────────────────────────┬──────────────────────────┘
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────┐
       │ Learned PyTorch Scorer (Deep MLP Neural Network)   │
       │ Linear(1537->128) -> BatchNorm -> ReLU -> Dropout  │
       │ Linear(128->32)   -> BatchNorm -> ReLU -> Dropout  │
       │ Linear(32->1)     -> Sigmoid -> Calibrated Score   │
       └─────────────────────────┬──────────────────────────┘
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────┐
       │ Streamlit UI Dashboard                             │
       │ 0-100% Calibrated Score | Green/Red Skill Badges   │
       │ Baseline Cosine vs Learned Scorer Comparison       │
       └────────────────────────────────────────────────────┘
```

---

## 🔬 3. Dataset & Model Training Pipeline

* **Dataset Source:** [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) from Hugging Face (thousands of real corporate job descriptions paired with actual multi-page candidate resumes).
* **Data Splits:** Strictly stratified **70% Train** (420 pairs), **15% Validation** (90 pairs), and **15% Test** (90 pairs).
* **Cross-Feature Engineering ($1537$ dimensions):**
  - Base vectors: Resume embedding $u \in \mathbb{R}^{384}$ and JD embedding $v \in \mathbb{R}^{384}$.
  - Element-wise distance: $|u - v| \in \mathbb{R}^{384}$.
  - Element-wise product: $u \odot v \in \mathbb{R}^{384}$.
  - Domain skill overlap ratio: $\text{overlap} \in \mathbb{R}^{1}$.
* **Regularization Strategy:**
  - Optimizer: `AdamW` ($\text{lr} = 3 \times 10^{-4}$, weight decay $= 0.01$).
  - Dropout: $0.40$ on Layer 1, $0.30$ on Layer 2.
  - Early stopping checkpointing saved based on minimum validation loss.

---

## 📊 4. Quantitative Experimental Results (Unseen Test Set)

The learned model was evaluated against the **Baseline Cosine Similarity** on 90 completely held-out test samples:

| Evaluation Metric | Baseline Cosine Similarity | Learned Scorer (PyTorch MLP) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Classification Accuracy** | 64.44% | **80.00%** | **+15.56%** |
| **Separation Gap (Match vs Mismatch)** | 11.27% | **26.74%** | **+15.47% (2.4x)** |
| **Ranking Quality (NDCG@5)** | 88.69% | **93.13%** | **+4.44%** |

### Evaluation Visualizations
- **Loss Curve:** Smooth, monotonically decreasing training and validation curves without overfitting.
- **NDCG@5 & Accuracy Chart:** Pronounced separation gap and ranking superiority over the baseline.

*(Artifacts saved in [`reports/loss_curve_regularized.png`](file:///home/praveen/Desktop/Sem-07/Deep-Learning/HireSync/reports/loss_curve_regularized.png) and [`reports/evaluation_comparison.png`](file:///home/praveen/Desktop/Sem-07/Deep-Learning/HireSync/reports/evaluation_comparison.png))*

---

## ⚖️ 5. Bias Risk Analysis & Mitigation (Syllabus Requirement)

### A. Bias Risks in Automated Resume Matching
1. **Keyword-Stuffing Exploits (Lexical Bias):**
   Traditional ATS algorithms over-weight exact keyword frequencies. A candidate who pastes a hidden block of keywords ("Python Python Python AWS Docker") can artificially score 90%+, while a candidate describing real engineering impact without repeating exact buzzwords gets filtered out.
2. **Vocabulary & Morphological Mismatch:**
   Candidates using alternative phrasing (e.g. *"pytest test suites"* vs *"quality assurance testing"*, or *"FastAPI microservices"* vs *"backend web development"*) face penalty under strict keyword parsers.
3. **Experience Overqualification vs Role Scope:**
   Senior systems engineers applying to specialized roles (e.g., distributed systems engineer applying to WordPress theme maintenance) may match general web keywords (HTML, JS, MySQL) but lack the specific CMS domain focus.

### B. HireSync Mitigation Strategies
1. **Dense Contextual Embeddings (Sentence-BERT):**
   By encoding complete sentences rather than isolated tokens, the model evaluates *semantic context*. It understands that *"architected an event-driven task pipeline with Redis and BullMQ"* implies backend scalability even if the words *"scalable backend"* are not explicitly typed.
2. **Global Semantic Soft-Matching ($\text{sim} \ge 0.75$):**
   Skills are extracted not only by strict string boundaries, but also cross-checked via SBERT token embeddings. If a candidate writes `"tests"`, the model computes $\text{Cosine}(\text{SBERT}("testing"), \text{SBERT}("tests")) = 0.921$ and marks it matched.
3. **Dual-Signal Hybrid Re-Ranking:**
   By feeding both dense semantic vectors ($u, v, |u-v|, u \odot v$) AND explicit skill overlap into the MLP, the model balances high-level semantic relevance with tangible required skills, preventing pure keyword stuffing from tricking the scorer.

---

## 📁 6. Repository Structure

```text
HireSync/
├── .gitignore
├── requirements.txt
├── README.md
├── models/
│   ├── learned_scorer.pth         # Trained PyTorch weights (~812 KB)
│   ├── model_config.json          # Network parameters & test metrics
│   └── skills_taxonomy.json       # 97 multi-domain skills dictionary
├── reports/
│   ├── loss_curve_regularized.png # Regularized training & validation loss curves
│   └── evaluation_comparison.png  # NDCG@5 & Accuracy comparison plot
├── src/
│   ├── __init__.py
│   ├── extractor.py               # pypdf parsing and text normalization
│   ├── skills.py                  # Global SBERT semantic soft-matching & gap finder
│   └── matcher.py                 # SBERT inference, feature synthesis & calibrated scoring
└── app.py                         # Interactive Streamlit application
```

---

## 🚀 7. Quick Start & Local Setup

### Prerequisites
* Python 3.10 or higher
* `pip` and `virtualenv`

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/Praveen1116/HireSync.git
cd HireSync

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**:
1. Upload any candidate resume (PDF or paste plain text).
2. Paste any target Job Description.
3. Click **"Analyze Match & Skill Gap"** to view:
   - **Learned Match Score** (0–100%) with qualitative badge (*Strong Match*, *Potential Match*, *Low Match*).
   - **Baseline Cosine Similarity** comparison.
   - **Matched Skills** (in **Emerald Green**).
   - **Missing Skills Gap** (in **Crimson Red**).
   - Academic evaluation tabs with NDCG@5 charts.

---

## 📜 8. License & Academic Attribution
Developed as part of the **Continuous Assessment (CA) Project** for **Deep Learning (SIH26044)**.  
Released under the MIT License.

