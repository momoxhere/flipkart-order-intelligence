# Flipkart Order Intelligence & Support Assistant

## Project Overview
This repository contains a connected, end-to-end intelligence system for Flipkart's catalog and customer-support teams. Instead of isolated scripts, the project integrates machine learning and Generative AI into a single LangGraph-based support assistant. 

The system consists of three connected parts:
1. **Return-Risk Prediction:** A machine learning pipeline trained on synthetic order history to predict the probability of an order being returned.
2. **Product-Image Categorisation:** A transfer-learning computer vision model built on Fashion-MNIST to classify product images.
3. **LangGraph Support Agent:** A deterministic, offline `MOCK_LLM` agent that answers policy queries via a grounded RAG knowledge base, seamlessly executing the saved models from Parts 1 and 2 as tools while maintaining conversational state and blocking prompt injections.

---

## Installation

Ensure you have Python 3.9+ installed. Set up a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
Part 1: Return-Risk Scoring PipelineRun Instructions:Bashpython generate_orders.py
python part1/train_return_risk.py
Data Verification & MissingnessTotal Rows: 6000 | Columns: 13Overall Return Rate: ~22.75%Missing rating_given: ~12.5%Missingness Classification: MAR (Missing At Random). Missingness systematically depends on the observed payment_method. For COD orders, missingness is ~22%, while for non-COD it is ~6%.Baseline & The High-Accuracy TrapA DummyClassifier predicting the majority class ("not returned") achieves ~77.2% accuracy but an F1-score of 0.00 for returns. Relying purely on accuracy is a business trap in imbalanced datasets; the model can look highly accurate while successfully catching exactly zero actual returns.Logistic Regression Threshold Trade-offThe optimal Logistic Regression threshold was found at 0.44 (maximizing F1 at 0.4091). Lowering the threshold from the default 0.50 catches more true returns (Recall increases), but flags more false positives (Precision decreases). Support teams must balance the cost of reviewing false positives against the savings of intercepting genuine returns.Feature Importance & Impurity BiasThe top five features based on impurity were payment_method_COD, price_inr, delivery_distance_km, customer_tenure_days, and delivery_days.Impurity Bias: When testing via Permutation Importance on held-out data, delivery_distance_km dropped substantially from ~0.095 (impurity) to near zero (~-0.0002). Impurity metrics inherently overrate noisy, continuous variables because they offer trees countless possible split points, inflating their apparent value even when true predictive power is weak.Subgroup AnalysisEvaluation across subgroups revealed a significant weakness based on payment_method. While COD recall was extremely high (0.9355), digital payments like Prepaid_Card (0.0204), Prepaid_UPI (0.0417), and Wallet (0.0952) showed drastically worse recall.Intervention: Implement payment-specific threshold calibrations, setting a more sensitive risk threshold specifically for Prepaid and Wallet orders.Return-Risk ResultsModelThresholdPrecisionRecallF1ROC-AUCDummyN/A0.0000.0000.0000.500Logistic Regression0.500.4680.2850.3540.612Logistic Regression0.440.4120.4070.4090.612Random Forestt*_rf (0.36)0.4510.5280.4860.654Random Forest TuningBest n_estimators: 200Best max_depth: 6CV ROC-AUC: 0.658Test ROC-AUC: 0.654 (Difference within 0.05)Part 2: Product Image CategoriserRun Instructions:Bashpython part2/train_classifier.py
python part2/predict_image.py
Architecture & StrategyDataset: Fashion-MNIST (60k train/10k test).Splits: 55,000 Training | 5,000 Validation (Stratified via scikit-learn) | 10,000 Test.Preprocessing: Grayscale converted to 3 channels, resized to 224x224, normalized to ImageNet statistics.Transfer Learning: Used a pretrained ResNet-18 backbone. Early/middle layers were frozen.Feature Caching: Features were extracted and cached via the frozen backbone prior to training the classification head, massively reducing CPU training time.Fine-tuning: Fine-tuning deeper layers was not required as validation accuracy exceeded 80% on the classification head alone.Image Classifier ResultsMetricValueTrain images55,000Validation images (Stratified)5,000Test images10,000Feature-extraction validation accuracy87.58%Fine-tuned validation accuracyN/A (Goal met without fine-tuning)Test Accuracy88.24%Confusion Matrix AnalysisShirt vs. T-shirt/top: Highly confused because both are upper-body garments with visually similar sleeves and necklines in low-resolution 28x28 grayscale.Coat vs. Pullover: Confused frequently as both are bulky, long-sleeved winter items lacking distinct structural outlines in low-fidelity representations.Part 3: LangGraph Support AgentRun Instructions:Bashpython part3/build_index.py
python -m part3.evaluate
Architecture & GuardrailsRAG Knowledge Base: 12 policy documents, chunked sentence-wise, embedded with all-MiniLM-L6-v2. FAISS was configured for Inner Product (IndexFlatIP) with normalized embeddings to enforce strict Cosine Similarity.Tools: Seamlessly loads models/return_risk_model.pkl and models/product_classifier.pt. Risk buckets are dynamically calibrated against t*_rf = 0.36.Agent State & Few-Shot Intent: Multi-turn interactions pass a real context state (tracking order IDs and image paths). Intent routing is governed explicitly by deterministic few-shot keyword rules.Guardrails:Prompt Injection: Blocks commands like "ignore previous instructions".Groundedness: Refuses policy questions if the Cosine Similarity falls below 0.40.Retrieval Evaluation (Document-Level)QueryRelevant DocsRetrieved DocsP@3R@3'How long do I have to return footwear?'POL002POL004, POL001, POL0020.3331.000'When will I get my COD refund?'POL005POL006, POL005, POL0080.3331.000'I received a broken laptop, what do I do?'POL003, POL010POL003, POL010, POL0110.6671.000'Can I return the lipstick I just opened?'POL012POL012, POL011, POL0010.3331.000'My prepaid refund hasn't arrived yet.'POL006POL006, POL005, POL0080.3331.000Average0.4001.000Example Transcript: Ungrounded Policy Refusal (Spaceship Test)(See all 8 full conversational transcripts in the transcripts/ directory)Markdown# Test 8

**User:** What is the policy for returning a spaceship?

**Agent JSON Response:**
```json
{
  "answer": "REFUSE. Top retrieved similarity: 0.1245. Grounding threshold: 0.4. I cannot answer ungrounded policy questions.",
  "source": "policy_kb",
  "confidence": 0.0
}


