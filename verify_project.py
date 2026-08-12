from pathlib import Path
import pandas as pd
import joblib
import json

assert Path("orders_dataset.csv").exists(), "orders_dataset.csv is missing"
assert Path("models/return_risk_model.pkl").exists(), "models/return_risk_model.pkl is missing"
assert Path("models/product_classifier.pt").exists(), "models/product_classifier.pt is missing"

print("Found required files.")

df = pd.read_csv("orders_dataset.csv")
assert df.shape == (6000, 13), f"orders_dataset.csv shape is {df.shape}, expected (6000, 13)"

print("orders_dataset.csv has expected shape.")

model = joblib.load("models/return_risk_model.pkl")
assert hasattr(model, "predict_proba"), "Loaded return risk model does not support predict_proba"

print("Return risk model loads and supports predict_proba.")

with open("part1/results/rf_threshold.json") as f:
    threshold = json.load(f)

assert 0.1 <= threshold["t_star_rf"] <= 0.9, f"t_star_rf is {threshold['t_star_rf']}, expected between 0.1 and 0.9"

print("RF threshold file loaded and value is within expected range.")

sample_images = list(Path("data/sample_images").glob("*.png"))
assert len(sample_images) >= 5, f"Found {len(sample_images)} sample images, expected at least 5"

print("Found sufficient sample images.")

print("ALL CORE ACCEPTANCE CHECKS PASSED")