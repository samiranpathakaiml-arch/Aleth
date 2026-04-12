---
title: Aleth
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Aleth 🔍

**v1.1.0** — OpenEnv-compliant benchmark for scientific citation verification and drift detection.

> *Aleth* (Greek: ἀλήθεια, *truth*) evaluates the ability of frontier LLM agents to verify whether academic citations actually support the claims made about them. Unlike standard RAG tasks, Aleth introduces **Adversarial Citation Drift** — where claims subtly manipulate source material (e.g., turning correlation into causation).

---

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Action Space](#action-space)
- [Task Hierarchy](#task-hierarchy)
- [Reward & Grading](#reward--grading)
- [Baseline Performance](#baseline-performance)
- [Docker Deployment](#docker-deployment)
- [License](#license)

---

## Overview

Aleth is designed to stress-test LLM agents on a task that requires genuine reading comprehension and logical reasoning: given a claim and a cited paper, does the paper actually support the claim?

The benchmark escalates across three difficulty tiers — from direct data matching to adversarial drift detection — making it suitable for evaluating both retrieval accuracy and higher-order reasoning capabilities.

---

## Directory Structure

```
aleth/                        # Project root
├── main.py                   # FastAPI server (OpenEnv REST endpoints)
├── environment.py            # AlethEnv (reset / step / state logic)
├── models.py                 # Pydantic schemas (Action, Observation, Reward, State)
├── grader.py                 # Deterministic grader (accuracy + reasoning + drift)
├── reward.py                 # Dense shaped reward engine
├── inference.py              # Baseline inference script (OpenAI-compatible)
├── openenv.yaml              # Official environment specification
├── requirements.txt          # Python dependencies
├── Dockerfile                # Containerised execution
├── server/
│   └── app.py               # Alternative entry point (uvicorn server.app:app)
└── data/                     # Benchmark task data
    ├── task_easy.json        # 5 NLP Landmark claims (direct verification)
    ├── task_medium.json      # 15 claims (multi-paper synthesis, abstract-only)
    └── task_hard.json        # 30 adversarial claims (citation drift)
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Aleth supports any OpenAI-compatible provider (Hugging Face Router, Anthropic, Groq). Set your variables before running:

**PowerShell:**
```powershell
$env:HF_TOKEN    = "your_token_here"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
```

**Bash / Zsh:**
```bash
export HF_TOKEN="your_token_here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
```

### 3. Run Baseline

```bash
python inference.py
```

> **Note:** The baseline script includes state-aware pruning to prevent context-length overflow.

---

## Action Space

| Action | Description | Use Case |
|---|---|---|
| `read_paper` | Returns paper text (truncated to 1k chars for efficiency) | Gathering evidence |
| `verify_claim` | Submit `support_score` [0.0–1.0] and `reasoning` | Core task completion |
| `flag_drift` | Provide `citation_chain` and `drift_explanation` | Required for Hard tasks |
| `submit` | Terminates the episode for grading | Budget management |

---

## Task Hierarchy

| Task | Claims | Complexity | Key Challenge |
|---|---|---|---|
| **Easy** | 5 | Direct | Exact data matching (e.g., BERT/GPT-3 parameters) |
| **Medium** | 15 | Synthesis | Multi-paper verification & abstract-only access |
| **Hard** | 30 | Adversarial | Citation Drift — detecting subtle logic manipulation |

---

## Reward & Grading

### Dense Rewards (per step)

| Event | Reward |
|---|---|
| Relevant read | `+0.02` |
| Verification accuracy | up to `+0.50` |
| Reasoning bonus (keyword overlap) | `+0.10` |
| Irrelevant read | `-0.05` |

### Final Grade Formula

Each claim is graded out of **1.0** using the following weighted formula:

$$\text{Score} = 0.60 \times \text{Accuracy} + 0.30 \times \text{Reasoning} + 0.10 \times \text{Drift}$$

| Component | Weight | Description |
|---|---|---|
| Accuracy | 60% | Correct support score for the claim |
| Reasoning | 30% | Quality of the agent's written justification |
| Drift | 10% | Detection of adversarial citation manipulation |

---

## Baseline Performance

| Task | Expected Range | Model | Notes |
|---|---|---|---|
| **Easy** | `0.70 – 0.90` | `meta-llama/Llama-3.1-8B-Instruct` | Reproducible via `inference.py` |
| **Medium** | `0.50 – 0.75` | `meta-llama/Llama-3.1-8B-Instruct` | Multi-paper synthesis |
| **Hard** | `0.30 – 0.60` | `meta-llama/Llama-3.1-8B-Instruct` | Adversarial citation drift |

Run the full baseline yourself:
```bash
# 1. Start the server
uvicorn main:app --host 0.0.0.0 --port 7860

# 2. In a separate terminal, run the baseline
export HF_TOKEN="your_token"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
python inference.py
```

---

## Docker Deployment

For Hugging Face Spaces or local containerization:

```bash
docker build -t aleth:latest .
docker run -e HF_TOKEN=$env:HF_TOKEN aleth:latest
```

> **Note:** Ensure your Hugging Face Space is tagged with `openenv` for automated leaderboard tracking.

---

## License

[MIT License](LICENSE) — Built for the **OpenEnv Hackathon 2026**.