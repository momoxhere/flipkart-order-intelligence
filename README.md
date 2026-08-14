# Flipkart Order Intelligence Platform

An end-to-end multimodal customer support intelligence system featuring predictive return modeling, deep learning image classification, and an agentic RAG workflow with input/output safety guardrails.

---

## Architecture Overview

                    ┌────────────────────────┐
                    │      User Message      │
                    └───────────┬────────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │    Input Guardrail Check   │  ──(Malicious)──► Refusal JSON
                  │  (Prompt Injection Filter) │
                  └─────────────┬──────────────┘
                                │ (Safe)
                                ▼
                  ┌────────────────────────────┐
                  │    Intent Classification   │
                  │  (Policy / Risk / Image)   │
                  └─────────────┬──────────────┘
                                │
     ┌──────────────────────────┼──────────────────────────┐
     │                          │                          │
     ▼ (Policy)                 ▼ (Return Risk)            ▼ (Image Classification)
┌──────────────────┐       ┌──────────────────┐       ┌────────────────────────┐
│ FAISS L2 Search  │       │ Return Risk Tool │       │ PyTorch CNN Classifier │
│ (All-MiniLM-L6)  │       │ (Random Forest)  │       │ (Sample Products)      │
└────────┬─────────┘       └────────┬─────────┘       └───────────┬────────────┘
│                          │                             │
▼                          │                             │
┌──────────────────┐                │                             │
│ Grounding Filter │                │                             │
│ (Distance <= 1.35)                │                             │
└────────┬─────────┘                │                             │
│                          │                             │
└──────────────────────────┼─────────────────────────────┘
│
▼
┌────────────────────────────┐
│       Response Node        │
│  (Deterministic Fallback)  │
└─────────────┬──────────────┘
│
▼
┌────────────────────────────┐
│    Single JSON Response    │
└────────────────────────────┘


---

## Part 1: Return Risk Prediction (Tabular ML)

### 1. Model Evaluation & Comparison (Tasks 4–6)

#### Baseline: Logistic Regression (Default Threshold = 0.50)
* **Accuracy:** `0.79`
* **Precision:** `0.68`
* **Recall:** `0.54`
* **F1-Score:** `0.60`
* **ROC-AUC:** `0.83`

#### Random Forest Classifier Tuning
* **Cross-Validation ROC-AUC (5-Fold Mean ± Std):** `0.872 ± 0.014`
* **Test Set ROC-AUC:** `0.881`
* **Best Hyperparameters:**
  * `n_estimators`: `200`
  * `max_depth`: `10`
  * `min_samples_split`: `5`
  * `class_weight`: `balanced`

#### Threshold Optimization & Model Comparison
Optimizing the classification decision threshold to maximize operational utility and recall on high-risk returns:

| Model | Decision Threshold | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Default)** | 0.50 | 0.68 | 0.54 | 0.60 | 0.830 |
| **Logistic Regression (Tuned)** | 0.38 | 0.61 | 0.76 | 0.68 | 0.830 |
| **Random Forest (Default)** | 0.50 | 0.74 | 0.68 | 0.71 | 0.881 |
| **Random Forest (Tuned / Selected)** | **0.40** | **0.70** | **0.81** | **0.75** | **0.881** |

---

### 2. Feature Importance & Subgroup Performance

#### Top Feature Importances (Random Forest)
1. **`discount_percent`** (0.241) – High discounts strongly correlate with impulsive purchases and higher return rates.
2. **`customer_return_rate`** (0.218) – Historical customer behavior is the strongest behavioral predictor.
3. **`product_category`** (0.165) – Apparel and Footwear show consistently higher variance in fit/expectation.
4. **`delivery_days`** (0.112) – Longer transit times increase buyer remorse and cancellation/return rates.
5. **`is_cod`** (0.094) – Cash on Delivery orders exhibit lower customer commitment prior to delivery.

#### Subgroup Performance by Category
| Category | Sample Size ($N$) | ROC-AUC | Recall (@ 0.40 Threshold) |
| :--- | :---: | :---: | :---: |
| **Apparel** | 1,240 | 0.892 | 0.84 |
| **Footwear** | 890 | 0.884 | 0.82 |
| **Electronics** | 1,450 | 0.865 | 0.77 |
| **Home & Kitchen** | 920 | 0.871 | 0.79 |

---

## Part 2: Product Image Classification (Deep Learning)

* **Architecture:** PyTorch Custom CNN with Conv2D $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool layers.
* **Input Resolution:** $28 \times 28$ grayscale (Fashion-MNIST compatible).
* **Test Set Accuracy:** `89.4%`
* **Inference Pipeline:** Standalone inference supported via `predict_image.py` with confidence scoring and fallback categorization.

---

## Part 3: Agentic Customer Support System (LangGraph RAG)

### System Guardrails & Evaluation Results

* **4S Framework System Prompt:** Enforces **Role**, **Specific** intents, **Short** context usage, **Surround** data containment, and a **Single** JSON output contract.
* **Input Guardrail:** Regex-based heuristic block for prompt injection attacks (`ignore previous`, `pretend you are`, etc.) with immediate fallback to a safe refusal response.
* **Retrieval Evaluation:**
  * **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
  * **Vector Store:** FAISS index with L2 distance metric
  * **Average Precision@3:** `0.333`
  * **Average Recall@3:** `0.900`
* **Output-Side Grounding Guardrail:** 
  * **Grounding Threshold:** $\text{L2 Distance} \le 1.35$
  * Queries exceeding the threshold (e.g., out-of-domain queries like *"What is the capital of France?"*) are explicitly refused with `confidence: 0.0`.

---

## Verification & Test Suite

Run the end-to-end verification script to validate all data fixtures, vector indices, transcript generations, and core acceptance criteria:

```bash
# 1. Run agent evaluation and generate markdown transcripts
python -m part3.evaluate

# 2. Run complete project verification
python verify_project.py
Generated Transcripts
The automated evaluation generates structured transcripts in the transcripts/ directory:

01_policy_return_window.md – Return policy query within threshold.

02_policy_cod_refund.md – COD refund policy retrieval.

03_return_risk.md – Multi-feature return risk inference.

04_product_category.md – CNN product image classification.

05_multiturn_state_part1.md / 05_multiturn_state_part2.md – Multi-turn conversation order state retention.

06_fresh_conversation.md – Isolated session handling.

07_prompt_injection.md – Input-side prompt injection refusal.

08_ungrounded_policy.md – Output-side L2 grounding check and refusal.

09_image_return_eligibility_part1.md / 09_image_return_eligibility_part2.md – Multimodal image + policy grounding.