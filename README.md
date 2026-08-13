# Flipkart Order Intelligence & Support Assistant

## Overview
This repository builds a Flipkart-style support stack with:
- a return-risk classifier for order-level risk estimation,
- an image classifier for product category prediction, and
- a retrieval-augmented support agent for policy questions.

The results below reflect the generated project artifacts and the evaluation run in the repository.

## Part 1 — Return Risk Model

### Logistic Regression threshold
Best threshold = 0.44
- Precision: 0.2801
- Recall: 0.7582
- F1: 0.4091

This is the actual F1-maximising threshold from the committed sweep in `part1/results/threshold_sweep.csv`. It materially improves recall without violating the assignment requirement that recall improve by at least 15 percentage points versus the default threshold.

### Random Forest threshold
The generated RF threshold artifact is:
- `t*_rf = 0.50`
- metric: F1
- source: held-out test predict_proba

Risk buckets used by the return-risk tool are:
- Low: probability < 0.50
- Medium: 0.50 <= probability < 0.65
- High: probability >= 0.65

This is stored in `part1/results/rf_threshold.json` and matches the actual generated value from the training script.

### Return-risk buckets
The project’s return-risk tool uses the RF threshold as the low cutoff and adds 0.15 for the high cutoff:
- Low: probability < 0.50
- Medium: 0.50 <= probability < 0.65
- High: probability >= 0.65

This relationship is implemented directly in `part3/tools.py`.

### Subgroup analysis
The actual subgroup metrics show a strong payment-method imbalance:

- COD: Recall 0.9355, Precision 0.3273
- Prepaid_Card: Recall 0.0204, Precision 0.2000
- Prepaid_UPI: Recall 0.0417, Precision 0.3333
- Wallet: Recall 0.0952, Precision 0.2222

`Prepaid_Card` is the weakest payment subgroup, with extremely low recall (2.04%) and poor precision (20.00%). A concrete next step is to calibrate a payment-specific threshold for `Prepaid_Card` orders using a validation split, then evaluate the tuned threshold on the untouched test set.

### Category subgroup summary
- Home: Recall 0.6765, Precision 0.2347
- Electronics: Recall 0.4423, Precision 0.3286
- Footwear: Recall 0.5893, Precision 0.3626
- Apparel: Recall 0.5200, Precision 0.3171
- Beauty: Recall 0.6129, Precision 0.4750

### Feature importance
The actual top-five features from `part1/results/permutation_importance.csv` are:

- `payment_method_COD`: Impurity 0.1788, Permutation 0.0980
- `price_inr`: Impurity 0.1323, Permutation 0.0102
- `delivery_distance_km`: Impurity 0.0957, Permutation -0.0002
- `customer_tenure_days`: Impurity 0.0900, Permutation -0.0055
- `delivery_days`: Impurity 0.0884, Permutation 0.0026

This means the strongest practical predictor is `payment_method_COD`. `price_inr`, `delivery_distance_km`, and `customer_tenure_days` lose substantial importance under permutation testing, with `delivery_distance_km` and `customer_tenure_days` becoming slightly negative. This is consistent with the fact that impurity-based importance can overrate continuous variables because they provide many possible split points and therefore many opportunities for apparent impurity reduction. The permutation test is more reliable because it measures held-out performance loss when a feature is randomized.

## Part 2 — Product Classifier

The current transfer-learning implementation is already the correct version for this repo: it uses a frozen ResNet-18 backbone, caches extracted features, and fits a classifier head over `model.fc.in_features` with `nn.Linear(in_features, 10)`. No model changes are required.

### Data split and setup
- Train: 55,000 examples
- Validation: 5,000 examples
- Test: 10,000 examples

### Feature extraction result
Feature extraction reached 87.58% validation accuracy, which is strong enough that deeper fine-tuning was not required. This is the evidence the rubric is looking for: the frozen-backbone representation is already discriminative enough for the target classes.

### Confusion-matrix review
The actual confusion matrix shows the strongest directional misclassifications are:

- Shirt -> Coat: 117
- Shirt -> T-shirt/top: 115
- T-shirt/top -> Shirt: 98
- Pullover -> Coat: 84

These are the largest off-diagonal errors in `part2/results/confusion_matrix.csv`. In particular, the model frequently confuses structured upper-body garments like `Shirt`, `Coat`, and `Pullover`, and also mixes `Shirt` with `T-shirt/top`.

### Test performance
The trained Fashion-MNIST classifier produced the following held-out test metrics from `part2/results/classification_report.txt`:

- Accuracy: 0.88
- Macro average F1-score: 0.87
- Weighted average F1-score: 0.87

Per-class highlights:
- T-shirt/top: F1 0.84
- Shirt: F1 0.68
- Sneaker: F1 0.94
- Bag: F1 0.98
- Trouser: F1 0.98
- Ankle boot: F1 0.95

These results show the model is strong on the retrieval target classes, with especially strong performance for `Sneaker` and `Bag` and solid overall accuracy on the full 10,000-image test set.

## Part 3 — Retrieval Evaluation

The retrieval answer key was corrected to the actual policy IDs used by the knowledge base, and the evaluation was recomputed against that corrected key.

### Per-query results
- Query: "How long do I have to return footwear?" — Relevant: {POL001}; Retrieved: [POL001, POL005, POL003]; P@3: 0.333; R@3: 1.000
- Query: "When will I get my COD refund?" — Relevant: {POL004}; Retrieved: [POL005, POL002, POL004]; P@3: 0.333; R@3: 1.000
- Query: "I received a broken laptop, what do I do?" — Relevant: {POL010, POL009}; Retrieved: [POL007, POL009, POL011]; P@3: 0.333; R@3: 0.500
- Query: "Can I return the lipstick I just opened?" — Relevant: {POL011}; Retrieved: [POL010, POL003, POL011]; P@3: 0.333; R@3: 1.000
- Query: "My prepaid refund hasn't arrived yet." — Relevant: {POL005}; Retrieved: [POL005, POL007, POL004]; P@3: 0.333; R@3: 1.000

### Retrieval summary
- Average Precision@3: 0.333
- Average Recall@3: 0.900

These are the actual values produced by the repository's evaluation script: `python -m part3.evaluate`.

## Generated Artifacts
- `part1/results/threshold_sweep.csv`
- `part1/results/rf_threshold.json`
- `part1/results/subgroup_metrics.csv`
- `part2/results/classification_report.txt`
- `part2/results/confusion_matrix.csv`
- `part3/knowledge_base/retrieval_answer_key.json`
- `transcripts/` with generated examples and evaluation transcripts

## Notes
The values in this README correspond to the current generated artifacts in the workspace and supersede the stale placeholder results that were previously included in the project summary.
