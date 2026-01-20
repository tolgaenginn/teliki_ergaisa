from pathlib import Path
import json
import joblib
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

# Resolve paths safely
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
MOD_DIR = OUT_DIR / "models"
TAB_DIR = OUT_DIR / "tables"
PRED_DIR = OUT_DIR / "preds"

MOD_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

mat_path = PRED_DIR / "matrices.joblib"
assert mat_path.exists(), f"Missing matrices file: {mat_path}. Run step2_preprocess_and_merge.py first."

mat = joblib.load(mat_path)
X_train_t = mat["X_train_t"]
X_valid_t = mat["X_valid_t"]
y_train = mat["y_train"].astype(np.float32)
y_valid = mat["y_valid"].astype(np.float32)

# Determine feature count and choose SVD dim
n_features = X_train_t.shape[1]
svd_dim = min(128, max(2, n_features - 1))

print("Transformed feature count:", n_features)
print("Using TruncatedSVD n_components:", svd_dim)

svd = TruncatedSVD(n_components=svd_dim, random_state=42)

X_train = svd.fit_transform(X_train_t).astype(np.float32)
X_valid = svd.transform(X_valid_t).astype(np.float32)

# Torch setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_ten = torch.tensor(X_train, dtype=torch.float32)
y_train_ten = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_valid_ten = torch.tensor(X_valid, dtype=torch.float32)
y_valid_ten = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_ten, y_train_ten), batch_size=256, shuffle=True)
valid_loader = DataLoader(TensorDataset(X_valid_ten, y_valid_ten), batch_size=512, shuffle=False)

in_dim = X_train.shape[1]

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(in_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

def rmse_np(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))

best_rmse = float("inf")
best_state = None

# Train
for epoch in range(1, 21):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()

    model.eval()
    preds_list = []
    ys_list = []
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().reshape(-1)
            preds_list.append(pred)
            ys_list.append(yb.numpy().reshape(-1))

    preds = np.concatenate(preds_list)
    ys = np.concatenate(ys_list)

    rmse = rmse_np(ys, preds)
    print(f"Epoch {epoch:02d} | RMSE={rmse:.6f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

# Save best model + SVD
torch.save(best_state, MOD_DIR / "mlp_best.pt")
joblib.dump(svd, MOD_DIR / "svd_for_mlp.joblib")

with open(TAB_DIR / "dl_results.json", "w") as f:
    json.dump({"mlp_rmse": float(best_rmse), "svd_dim": int(svd_dim), "n_features_before_svd": int(n_features)}, f, indent=2)

print("\nBest MLP RMSE:", best_rmse)
print("Saved:")
print(" - outputs/models/mlp_best.pt")
print(" - outputs/models/svd_for_mlp.joblib")
print(" - outputs/tables/dl_results.json")
print("\nSTEP 5 COMPLETED SUCCESSFULLY.")
