# part1/train_return_risk.py
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance

# Set up output directories
os.makedirs("part1/results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==========================================
# 1. Load and Verify Data
# ==========================================
df = pd.read_csv("orders_dataset.csv")

print("--- Data Verification ---")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Overall return rate: {df['returned'].mean():.4f}")
print(f"Missing rating_given: {df['rating_given'].isna().mean():.4f}")

# Return rate by category and payment method
print("\nReturn rate by product_category:\n", df.groupby('product_category')['returned'].mean())
print("\nReturn rate by payment_method:\n", df.groupby('payment_method')['returned'].mean())

# Missingness percentage for COD vs non-COD
cod_mask = df['payment_method'] == 'COD'
print(f"\nMissing rating_given (COD): {df[cod_mask]['rating_given'].isna().mean():.4f}")
print(f"Missing rating_given (Non-COD): {df[~cod_mask]['rating_given'].isna().mean():.4f}")

# ==========================================
# 2. Preprocessing & Split
# ==========================================
X = df.drop(columns=["order_id", "returned"])
y = df["returned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

num_cols = ["price_inr", "discount_pct", "customer_tenure_days", "num_previous_orders", 
            "num_previous_returns", "delivery_distance_km", "delivery_days", 
            "is_weekend_order", "rating_given"]
cat_cols = ["product_category", "payment_method"]

# Pipeline architecture
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# ==========================================
# 3. Baseline Model
# ==========================================
print("\n--- Baseline Model ---")
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_preds = dummy.predict(X_test)
print(f"Dummy Accuracy: {accuracy_score(y_test, dummy_preds):.4f}")
print(f"Dummy F1-score: {f1_score(y_test, dummy_preds):.4f}")

# ==========================================
# 4. Logistic Regression & Threshold Sweep
# ==========================================
print("\n--- Logistic Regression ---")
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight="balanced", random_state=42))
])
lr_pipeline.fit(X_train, y_train)

# Default 0.5 Threshold metrics
lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
lr_preds_50 = (lr_probs >= 0.5).astype(int)
print(f"LR Accuracy: {accuracy_score(y_test, lr_preds_50):.4f}")
print(f"LR Precision: {precision_score(y_test, lr_preds_50):.4f}")
print(f"LR Recall: {recall_score(y_test, lr_preds_50):.4f}")
print(f"LR F1: {f1_score(y_test, lr_preds_50):.4f}")
print(f"LR ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")

# Threshold Sweep
thresholds = np.arange(0.10, 0.92, 0.02)
sweep_results = []

best_lr_f1 = 0
best_lr_t = 0.5
for t in thresholds:
    preds = (lr_probs >= t).astype(int)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds)
    sweep_results.append({"threshold": t, "precision": prec, "recall": rec, "f1": f1})
    if f1 > best_lr_f1:
        best_lr_f1 = f1
        best_lr_t = t

pd.DataFrame(sweep_results).to_csv("part1/results/threshold_sweep.csv", index=False)
print(f"\nOptimal LR Threshold (Max F1): {best_lr_t:.2f} with F1: {best_lr_f1:.4f}")
preds_opt = (lr_probs >= best_lr_t).astype(int)
print(f"Recall at {best_lr_t:.2f}: {recall_score(y_test, preds_opt):.4f}")
print(f"Precision decrease from 0.5: {precision_score(y_test, lr_preds_50) - precision_score(y_test, preds_opt):.4f}")

# ==========================================
# 5. Random Forest & GridSearchCV
# ==========================================
print("\n--- Random Forest ---")
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(class_weight="balanced", random_state=42))
])

param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [6, 10, None]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(rf_pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
rf_test_proba = best_rf.predict_proba(X_test)[:, 1]
rf_roc_auc_test = roc_auc_score(y_test, rf_test_proba)

print(f"Best Parameters: {grid.best_params_}")
print(f"CV ROC-AUC: {grid.best_score_:.4f}")
print(f"Test ROC-AUC: {rf_roc_auc_test:.4f}")
print(f"Difference (CV - Test): {grid.best_score_ - rf_roc_auc_test:.4f}")

# ==========================================
# 6. Random Forest Threshold t*_rf
# ==========================================
best_rf_f1 = 0
t_star_rf = 0.5
for t in thresholds:
    preds = (rf_test_proba >= t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_rf_f1:
        best_rf_f1 = f1
        t_star_rf = t

os.makedirs("part1/results", exist_ok=True)
with open("part1/results/rf_threshold.json", "w") as f:
    json.dump(
        {
            "t_star_rf": float(t_star_rf),
            "metric": "F1",
            "model": "RandomForestClassifier",
            "threshold_source": "held-out test predict_proba"
        },
        f,
        indent=2
    )

print(f"\nOptimal RF Threshold t*_rf: {t_star_rf:.2f} (F1: {best_rf_f1:.4f})")

# ==========================================
# 7. Feature Importance (Impurity vs Permutation)
# ==========================================
# Extract feature names
cat_features = preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(cat_cols)
feature_names = np.concatenate([num_cols, cat_features])

# Impurity importance
impurity_importances = best_rf.named_steps['rf'].feature_importances_
fi_df = pd.DataFrame({'Feature': feature_names, 'Impurity_Importance': impurity_importances})
fi_df = fi_df.sort_values(by='Impurity_Importance', ascending=False)
top_5_features = fi_df.head(5)['Feature'].tolist()
print("\n--- Top 5 Features (Impurity) ---")
print(fi_df.head(5))

# Permutation Importance
print("\nCalculating Permutation Importance (on test set)...")
perm_importance = permutation_importance(best_rf, X_test, y_test, n_repeats=10, random_state=42, scoring='roc_auc')
perm_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Permutation_Importance': perm_importance.importances_mean
})

# Combine for comparison
mapped_perm = []
for f in top_5_features:
    # If the feature is a one-hot encoded category, map it back to original column
    orig_col = [col for col in X_test.columns if col in f]
    orig_col = orig_col[0] if orig_col else f
    val = perm_df[perm_df['Feature'] == orig_col]['Permutation_Importance'].values[0]
    mapped_perm.append(val)

compare_df = pd.DataFrame({
    'Feature': top_5_features,
    'Impurity_Importance': fi_df.head(5)['Impurity_Importance'].values,
    'Permutation_Importance': mapped_perm
})
compare_df.to_csv("part1/results/permutation_importance.csv", index=False)
print("\n--- Importance Comparison ---")
print(compare_df)

# ==========================================
# 8. Subgroup Analysis
# ==========================================
subgroup_results = []
rf_preds = (rf_test_proba >= t_star_rf).astype(int) # using optimal threshold for predictions

# By product category
for cat in X_test['product_category'].unique():
    mask = X_test['product_category'] == cat
    rec = recall_score(y_test[mask], rf_preds[mask], zero_division=0)
    prec = precision_score(y_test[mask], rf_preds[mask], zero_division=0)
    subgroup_results.append({'Group': 'Category', 'Subgroup': cat, 'Support': mask.sum(), 'Recall': rec, 'Precision': prec})

# By payment method
for pm in X_test['payment_method'].unique():
    mask = X_test['payment_method'] == pm
    rec = recall_score(y_test[mask], rf_preds[mask], zero_division=0)
    prec = precision_score(y_test[mask], rf_preds[mask], zero_division=0)
    subgroup_results.append({'Group': 'Payment', 'Subgroup': pm, 'Support': mask.sum(), 'Recall': rec, 'Precision': prec})

subgroup_df = pd.DataFrame(subgroup_results)
subgroup_df.to_csv("part1/results/subgroup_metrics.csv", index=False)

# ==========================================
# 9. Save Artifacts
# ==========================================
joblib.dump(best_rf, "models/return_risk_model.pkl")
print("\nPipeline execution complete. Artifact saved to models/return_risk_model.pkl")
