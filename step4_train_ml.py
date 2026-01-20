from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Resolve paths
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
MOD_DIR = OUT_DIR / "models"
TAB_DIR = OUT_DIR / "tables"
PRED_DIR = OUT_DIR / "preds"

MOD_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

# Load matrices from Step 2
mat_path = PRED_DIR / "matrices.joblib"
assert mat_path.exists(), f"Missing matrices file: {mat_path}. Run step2_preprocess_and_merge.py first."

mat = joblib.load(mat_path)
X_train_t = mat["X_train_t"]
X_valid_t = mat["X_valid_t"]
y_train = mat["y_train"]
y_valid = mat["y_valid"]

results = {}

def rmse_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))

def fit_eval_save(name, model):
    model.fit(X_train_t, y_train)
    pred = model.predict(X_valid_t)
    rmse = rmse_score(y_valid, pred)
    results[name] = {"rmse": rmse}
    joblib.dump(model, MOD_DIR / f"{name}.joblib")
    return rmse

# 1) Ridge
fit_eval_save("ridge", Ridge(alpha=1.0))

# 2) Random Forest
fit_eval_save(
    "random_forest",
    RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
)

# 3) XGBoost
xgb_used = False
try:
    from xgboost import XGBRegressor
    xgb_used = True
    fit_eval_save(
        "xgboost",
        XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
    )
except Exception as e:
    results["xgboost_unavailable"] = {"error": str(e)}
    fit_eval_save("gradient_boosting_fallback", GradientBoostingRegressor(random_state=42))

# Pick best model
model_names = [k for k in results.keys() if "rmse" in results[k]]
best_name = min(model_names, key=lambda k: results[k]["rmse"])
best_path = str(MOD_DIR / f"{best_name}.joblib")

# Save results + best model info
with open(TAB_DIR / "ml_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open(MOD_DIR / "best_ml.json", "w") as f:
    json.dump({"best_model_name": best_name, "best_model_path": best_path}, f, indent=2)

print("ML results:")
for k in model_names:
    print(f" - {k}: RMSE={results[k]['rmse']:.6f}")

print("\nBest ML model:", best_name, "RMSE:", results[best_name]["rmse"])
print("Saved:")
print(" - outputs/tables/ml_results.json")
print(" - outputs/models/best_ml.json")
print("\nSTEP 4 COMPLETED SUCCESSFULLY.")
