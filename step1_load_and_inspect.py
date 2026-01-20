from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 160)

# Resolve paths
ROOT = Path(__file__).resolve().parents[1]   # project root
DATA_DIR = ROOT / "data"

assert DATA_DIR.exists(), f"Data directory not found: {DATA_DIR}"

print("Project root:", ROOT)
print("Data directory:", DATA_DIR)

# List files and detect datasets
files = list(DATA_DIR.glob("*.csv"))
print("\nCSV files found in data/:")
for f in files:
    print(" -", f.name)

train_features_file = None
train_labels_file = None
val_features_file = None

for f in files:
    name = f.name.lower()
    if "train" in name and "feature" in name:
        train_features_file = f
    elif ("train" in name and "gt" in name) or ("train" in name and "label" in name):
        train_labels_file = f
    elif "test" in name or "val" in name:
        val_features_file = f

assert train_features_file is not None, "Could not detect training FEATURES file"
assert train_labels_file is not None, "Could not detect training LABELS file"
assert val_features_file is not None, "Could not detect validation/test FEATURES file"

print("\nDetected files:")
print("Train features :", train_features_file.name)
print("Train labels   :", train_labels_file.name)
print("Val features   :", val_features_file.name)

# Load datasets
train_features = pd.read_csv(train_features_file)
train_labels = pd.read_csv(train_labels_file)
val_features = pd.read_csv(val_features_file)


# inspection
print("\nShapes:")
print("Train features :", train_features.shape)
print("Train labels   :", train_labels.shape)
print("Val features   :", val_features.shape)

print("\nTrain features head:\n", train_features.head(3))
print("\nTrain labels head:\n", train_labels.head(3))
print("\nVal features head:\n", val_features.head(3))

# Data types
print("\nTrain feature dtypes counts:\n")
print(train_features.dtypes.value_counts())

# Identify ID columns
common_cols = set(train_features.columns).intersection(set(train_labels.columns))
possible_id_cols = [c for c in common_cols if "id" in c.lower() or "hh" in c.lower()]

print("\nCommon columns (likely IDs):")
print(sorted(list(common_cols))[:30])

print("\nPossible ID columns:")
print(possible_id_cols)

# Identify target column
candidate_targets = [c for c in train_labels.columns if c not in train_features.columns]

print("\nCandidate target columns:")
print(candidate_targets)

if len(candidate_targets) == 1:
    target_col = candidate_targets[0]
    print("\nTarget column detected:", target_col)
    print("\nTarget statistics:\n", train_labels[target_col].describe())
else:
    print("\nWARNING: Could not uniquely infer target column. Check labels manually.")

# Missing values overview
missing_ratio = train_features.isnull().mean().sort_values(ascending=False)

print("\nTop 20 features by missing ratio:\n")
print(missing_ratio.head(20))

print("\nSTEP 1 COMPLETED SUCCESSFULLY.")
