from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

# Resolve paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
SUB_DIR = ROOT / "submissions"
MOD_DIR = OUT_DIR / "models"
PRED_DIR = OUT_DIR / "preds"

SUB_DIR.mkdir(exist_ok=True)

# Auto-detect CSV files
files = list(DATA_DIR.glob("*.csv"))

train_features_file = None
train_labels_file = None
val_features_file = None

for f in files:
    name = f.name.lower()
    if "train" in name and "feature" in name:
        train_features_file = f
    elif ("train" in name and "gt" in name) or ("train" in name and "label" in name):
        train_labels_file = f
    elif ("test" in name) or ("val" in name) or ("validation" in name):
        val_features_file = f

assert train_features_file is not None, "Training FEATURES file not found"
assert train_labels_file is not None, "Training LABELS file not found"
assert val_features_file is not None, "Validation/TEST FEATURES file not found"

print("Using files:")
print("Train features :", train_features_file.name)
print("Train labels   :", train_labels_file.name)
print("Val features   :", val_features_file.name)

# Load data
train_features = pd.read_csv(train_features_file)
train_labels = pd.read_csv(train_labels_file)
val_features = pd.read_csv(val_features_file)

# Load preprocessing + metadata
preprocessor = joblib.load(MOD_DIR / "preprocessor.joblib")
mat = joblib.load(PRED_DIR / "matrices.joblib")

merge_keys = mat["merge_keys"]
TARGET = mat["target"]
high_missing_cols = mat["high_missing_cols"]

# Merge train data
train = train_features.merge(train_labels, on=merge_keys, how="inner")
y_full = train[TARGET].astype(float).values

X_full = train.drop(columns=[TARGET])
X_full = X_full.drop(columns=merge_keys, errors="ignore")
X_full = X_full.drop(columns=high_missing_cols, errors="ignore")

X_val = val_features.drop(columns=merge_keys, errors="ignore")
X_val = X_val.drop(columns=high_missing_cols, errors="ignore")

# Transform
X_full_t = preprocessor.transform(X_full)
X_val_t = preprocessor.transform(X_val)

# Load best ML model
best_info = json.load(open(MOD_DIR / "best_ml.json"))
best_model_name = best_info["best_model_name"]
best_model_path = Path(best_info["best_model_path"])

best_model = joblib.load(best_model_path)

print("Using best model:", best_model_name)

# Train on full data
best_model.fit(X_full_t, y_full)

# Predict validation
val_pred = best_model.predict(X_val_t)

# Build submission
id_df_val = val_features[merge_keys].copy()

if len(merge_keys) == 1:
    id_col = merge_keys[0]
    submission = pd.DataFrame({
        id_col: id_df_val[id_col].values,
        TARGET: val_pred
    })
else:
    # Combine multiple keys into one ID column
    submission = id_df_val.astype(str).agg("_".join, axis=1).to_frame(name="row_id")
    submission[TARGET] = val_pred

out_path = SUB_DIR / "submission.csv"
submission.to_csv(out_path, index=False)

print("\nSUBMISSION FILE CREATED SUCCESSFULLY")
print("Saved to:", out_path)
print("Columns:", submission.columns.tolist())
