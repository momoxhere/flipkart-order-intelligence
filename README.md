# Flipkart Order Intelligence & Support Assistant

## Project Overview
This repository implements a connected intelligence pipeline for Flipkart-style order support. It includes a return-risk model, an image classifier, and a deterministic LangGraph-based support agent that combines policy retrieval with saved ML tools.

## Repository Structure
- `generate_orders.py` - Synthetic order generation for Part 1.
- `orders_dataset.csv` - Generated order dataset.
- `part1/` - Return-risk training and analysis.
- `part2/` - Product image classifier training and inference.
- `part3/` - Support agent, tools, knowledge base, and transcript generation.
- `models/` - Saved return-risk and image classification artifacts.
- `vector_index/` - FAISS index and metadata for policy retrieval.
- `transcripts/` - Generated example conversations and evaluation transcripts.

## Part 1 — Return Risk
### Dataset
Part 1 uses a synthetic order dataset with customer, delivery, and product features. The dataset includes return labels for training and validation.

### Preprocessing
Data preprocessing includes handling missing values, encoding categorical fields, and feature engineering for order and delivery patterns.

### Baseline
A majority-class baseline was evaluated to demonstrate the imbalance challenge in return prediction.

### Logistic Regression
A logistic regression model was trained and evaluated with a probability threshold calibration step.

### Random Forest
A random forest model was tuned, and the best hyperparameters were selected via cross-validation.

### Feature Importance
Feature importance was analyzed using impurity-based metrics to identify strong predictors.

### Permutation Importance
Permutation importance was used to measure the true contribution of each feature on held-out data.

### Subgroup Analysis
Performance was evaluated across customer subgroups, highlighting weaker performance for certain payment methods like Prepaid Card.

### t*_rf
The optimal random forest probability threshold `t*_rf` was identified and persisted for risk bucket calibration.

### Artifact
The trained return-risk model and threshold are saved in `models/return_risk_model.pkl` and `part1/results/rf_threshold.json`.

## Part 2 — Product Classifier
### Dataset
Part 2 uses Fashion-MNIST data to build a product category classifier for apparel images.

### Split
The dataset is split into 55,000 training, 5,000 validation, and 10,000 test images with stratified sampling.

### Preprocessing
Images are converted to 3-channel RGB, resized to 224×224, and normalized using ImageNet statistics.

### Transfer Learning
A pretrained ResNet-18 backbone is used, with only the classification head trained on the target categories.

### Feature Extraction
Features are extracted and cached from the frozen backbone to speed up training.

### Validation
A validation set is used to monitor performance and avoid overfitting during head training.

### Test Results
The final classifier is evaluated on the held-out Fashion-MNIST test set.

### Confusion Matrix
A confusion matrix is used to inspect common category confusions, such as between shirts and T-shirts/top.

### Sample Images
Sample product images are stored under `data/sample_images/` for demo inference.

### Artifact
The trained image classifier is saved to `models/product_classifier.pt`.

## Part 3 — Support Agent
### Architecture
Part 3 builds a LangGraph agent that routes queries to policy retrieval, return-risk tools, or image classification tools.

### Knowledge Base
The knowledge base contains policy documents in `part3/knowledge_base/policies.json` and is chunked for retrieval.

### RAG
Policy chunks are embedded and indexed with FAISS for retrieval. The agent performs grounded retrieval before answering policy questions.

### Tools
The agent uses saved artifacts from Part 1 and Part 2 to execute return-risk scoring and image classification.

### LangGraph
The graph has four nodes: `intent`, `retrieval`, `tool`, and `response`. It routes policy queries to retrieval and tool queries to the appropriate model execution.

### State
Conversation state persists order IDs, image paths, and tool outputs across turns. Fresh conversations reset this state.

### Guardrails
Prompt injection is blocked, and ungrounded policy queries are refused based on retrieval distance thresholds.

### MOCK_LLM
The default support agent mode is deterministic `MOCK_LLM`.
No API key is required.
No network connection is required.
No paid LLM service is required.

The default configuration is `USE_LIVE_LLM=0`.

## Running the Project
```bash
git clone <repo-url>
cd flipkart-order-intelligence
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Part 1
```bash
python3 generate_orders.py
python3 part1/train_return_risk.py
```

### Part 2
```bash
python3 part2/train_classifier.py
```

### Part 3
```bash
python3 part3/build_index.py
python3 part3/evaluate.py
```

## Retrieval Evaluation
The support agent includes retrieval evaluation for P@3 and R@3 on policy queries using document-level scoring.

## Test Transcripts
Generated transcripts are saved in the `transcripts/` directory, including policy, return risk, image classification, multi-turn state, fresh conversation reset, prompt injection blocking, ungrounded refusal, and mixed-intent examples.

## Git Workflow
Use feature branches and create pull requests for changes:
```bash
git checkout -b feature/<name>
git add .
git commit -m "Describe changes"
git push origin feature/<name>
```
Review changes before merging into `main`.
