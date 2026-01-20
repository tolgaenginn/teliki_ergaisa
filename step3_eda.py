from pathlib import Path
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Resolve paths safely
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
PRED_DIR = OUT_DIR / "preds"

FIG_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

# Load preprocessed matrices
mat_path = PRED_DIR / "matrices.joblib"
assert mat_path.exists(), f"Missing matrices file: {mat_path}"

mat = joblib.load(mat_path)

X_train_t = mat["X_train_t"]
X_valid_t = mat["X_valid_t"]
y_train = mat["y_train"]
y_valid = mat["y_valid"]

# Target distribution plot
plt.figure(figsize=(7, 4))
plt.hist(y_train, bins=40)
plt.title("Target Distribution (Train Split)")
plt.xlabel("Target value")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIG_DIR / "target_distribution.png", dpi=200)
plt.close()

# Quick baseline model for EDA
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_t, y_train)
preds = rf.predict(X_valid_t)

# RMSE computation
mse = mean_squared_error(y_valid, preds)
rmse = np.sqrt(mse)

# Save baseline result
with open(TAB_DIR / "eda_baseline.json", "w") as f:
    json.dump(
        {
            "random_forest_rmse": float(rmse)
        },
        f,
        indent=2
    )

print("EDA baseline RandomForest RMSE:", rmse)
print("Saved:")
print(" - outputs/figures/target_distribution.png")
print(" - outputs/tables/eda_baseline.json")
print("\nSTEP 3 COMPLETED SUCCESSFULLY.")
