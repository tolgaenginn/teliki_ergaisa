from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "models").mkdir(exist_ok=True)
(OUT_DIR / "preds").mkdir(exist_ok=True)
(OUT_DIR / "tables").mkdir(exist_ok=True)

assert DATA_DIR.exists(), f"Missing data directory: {DATA_DIR}"

#  CSV files
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

# Detect merge keys
common_cols = list(set(train_features.columns).intersection(set(train_labels.columns)))
assert len(common_cols) > 0, "No common columns to merge on"

id_cols = [c for c in common_cols if ("id" in c.lower()) or ("hh" in c.lower())]
merge_keys = id_cols if len(id_cols) > 0 else common_cols

print("Merge keys:", merge_keys)

# Detect target column
candidate_targets = [c for c in train_labels.columns if c not in train_features.columns]
assert len(candidate_targets) == 1, f"Ambiguous target columns: {candidate_targets}"

TARGET = candidate_targets[0]
print("Target column:", TARGET)

# Merge train data
train = train_features.merge(train_labels, on=merge_keys, how="inner")

y = train[TARGET].astype(float)
X = train.drop(columns=[TARGET])

# Preserve IDs for later submission
train_ids = X[merge_keys].copy()
val_ids = val_features[merge_keys].copy()

# Drop ID columns from modeling features
X = X.drop(columns=merge_keys, errors="ignore")
val_X = val_features.drop(columns=merge_keys, errors="ignore")

# Drop columns with too many missing values
missing_ratio = X.isnull().mean()
high_missing_cols = missing_ratio[missing_ratio > 0.85].index.tolist()

if high_missing_cols:
    print(f"Dropping {len(high_missing_cols)} high-missing columns (missing > 85%).")

X = X.drop(columns=high_missing_cols, errors="ignore")
val_X = val_X.drop(columns=high_missing_cols, errors="ignore")

# Column types
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"Numerical cols: {len(num_cols)} | Categorical cols: {len(cat_cols)}")

# Preprocessing pipeline
numeric_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_tf = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols)
    ],
    remainder="drop"
)

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Transform
X_train_t = preprocessor.fit_transform(X_train)
X_valid_t = preprocessor.transform(X_valid)
X_full_t = preprocessor.transform(X)
X_val_t = preprocessor.transform(val_X)

# Save artifacts
joblib.dump(preprocessor, OUT_DIR / "models" / "preprocessor.joblib")

joblib.dump(
    {
        "X_train_t": X_train_t,
        "X_valid_t": X_valid_t,
        "y_train": y_train.values,
        "y_valid": y_valid.values,
        "X_full_t": X_full_t,
        "y_full": y.values,
        "X_val_t": X_val_t,
        "merge_keys": merge_keys,
        "target": TARGET,
        "high_missing_cols": high_missing_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols
    },
    OUT_DIR / "preds" / "matrices.joblib"
)

train_ids.to_csv(OUT_DIR / "preds" / "train_ids.csv", index=False)
val_ids.to_csv(OUT_DIR / "preds" / "val_ids.csv", index=False)

with open(OUT_DIR / "tables" / "preprocess_meta.json", "w") as f:
    json.dump(
        {
            "target": TARGET,
            "merge_keys": merge_keys,
            "n_num_cols": len(num_cols),
            "n_cat_cols": len(cat_cols),
            "n_dropped_high_missing_cols": len(high_missing_cols),
        },
        f,
        indent=2
    )

print("\nSTEP 2 COMPLETED SUCCESSFULLY.")
print("Saved:")
print(" - outputs/models/preprocessor.joblib")
print(" - outputs/preds/matrices.joblib")
print(" - outputs/tables/preprocess_meta.json")
