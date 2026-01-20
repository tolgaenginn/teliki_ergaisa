from pathlib import Path
import json
import zipfile
import joblib
import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MOD_DIR = OUT_DIR / "models"
SUB_DIR = ROOT / "submissions"
SUB_DIR.mkdir(exist_ok=True)

# Auto-detect files
files = list(DATA_DIR.glob("*.csv"))
train_features_file = None
train_labels_file = None
test_features_file = None

for f in files:
    name = f.name.lower()
    if "train" in name and "feature" in name:
        train_features_file = f
    elif ("train" in name and "gt" in name) or ("train" in name and "label" in name):
        train_labels_file = f
    elif ("test" in name) or ("val" in name) or ("validation" in name):
        test_features_file = f

assert train_features_file is not None, "Training FEATURES file not found"
assert train_labels_file is not None, "Training LABELS file not found"
assert test_features_file is not None, "Test/validation FEATURES file not found"

print("Using files:")
print("Train features:", train_features_file.name)
print("Train labels  :", train_labels_file.name)
print("Test features :", test_features_file.name)

train_features = pd.read_csv(train_features_file)
train_labels = pd.read_csv(train_labels_file)
test_features = pd.read_csv(test_features_file)

# Detect keys + target

common_cols = list(set(train_features.columns).intersection(set(train_labels.columns)))
id_cols = [c for c in common_cols if ("id" in c.lower()) or ("hh" in c.lower())]
merge_keys = id_cols if len(id_cols) > 0 else common_cols
assert "survey_id" in merge_keys, f"Expected survey_id in merge keys, got: {merge_keys}"

hh_col = "hhid" if "hhid" in merge_keys else [c for c in merge_keys if c != "survey_id"][0]

candidate_targets = [c for c in train_labels.columns if c not in train_features.columns]
assert len(candidate_targets) == 1, f"Ambiguous target columns: {candidate_targets}"
TARGET = candidate_targets[0]

print("Merge keys:", merge_keys)
print("Household id column:", hh_col)
print("Target:", TARGET)

# Load preprocessing + best model
preprocessor = joblib.load(MOD_DIR / "preprocessor.joblib")

best_info = json.load(open(MOD_DIR / "best_ml.json"))
best_model = joblib.load(best_info["best_model_path"])
print("Best model:", best_info["best_model_name"])

# Train best model on training data
train = train_features.merge(train_labels, on=merge_keys, how="inner")
y_full = train[TARGET].astype(float).values

X_full = train.drop(columns=[TARGET]).drop(columns=merge_keys, errors="ignore")
X_test = test_features.drop(columns=merge_keys, errors="ignore")

X_full_t = preprocessor.transform(X_full)
X_test_t = preprocessor.transform(X_test)

best_model.fit(X_full_t, y_full)

test_pred = best_model.predict(X_test_t).astype(float)


hh_out = pd.DataFrame({
    "survey_id": test_features["survey_id"].values,
    "hhid": test_features[hh_col].values,
    "cons_ppp17": test_pred
})


hh_csv = SUB_DIR / "predicted_household_consumption.csv"
hh_out.to_csv(hh_csv, index=False)
print("Wrote:", hh_csv)


thresholds = [3.17, 3.94, 4.60, 5.26, 5.88, 6.47, 7.06, 7.70, 8.40,
              9.13, 9.87, 10.70, 11.62, 12.69, 14.03, 15.64, 17.76, 20.99, 27.37]

weight_candidates = [c for c in test_features.columns if "weight" in c.lower()]
weight_col = None
if len(weight_candidates) == 1:
    weight_col = weight_candidates[0]
elif len(weight_candidates) > 1:
    for preferred in ["weight", "hh_weight", "hhweight", "pop_weight", "popweight", "wgt"]:
        for c in weight_candidates:
            if preferred == c.lower():
                weight_col = c
                break
        if weight_col is not None:
            break

if weight_col is None:
    print("WARNING: No weight column confidently detected. Using equal weights (unweighted rates).")
    weights = np.ones(len(test_features), dtype=float)
else:
    print("Using weight column:", weight_col)
    weights = test_features[weight_col].astype(float).values
    # guard against zeros/negatives
    weights = np.where(weights > 0, weights, 0.0)

surveys = sorted(test_features["survey_id"].unique().tolist())

rows = []
for sid in surveys:
    mask = (test_features["survey_id"].values == sid)
    preds_s = test_pred[mask]
    w_s = weights[mask]
    denom = w_s.sum()
    if denom <= 0:
        w_s = np.ones_like(preds_s, dtype=float)
        denom = w_s.sum()

    row = {"survey_id": sid}
    for t in thresholds:
        col = f"pct_hh_below_{t:.2f}"
        row[col] = float(w_s[preds_s < t].sum() / denom)  # strictly below
    rows.append(row)

poverty_out = pd.DataFrame(rows)
poverty_csv = SUB_DIR / "predicted_poverty_distribution.csv"
poverty_out.to_csv(poverty_csv, index=False)
print("Wrote:", poverty_csv)

# Zip both CSVs at root
zip_path = SUB_DIR / "submission.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(hh_csv, arcname=hh_csv.name)
    z.write(poverty_csv, arcname=poverty_csv.name)

print("\nDONE")
print(zip_path)
