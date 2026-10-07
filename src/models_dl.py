"""
Module M3b - Deep-Learning Demand Forecaster (LSTM).

Implements the deep-learning member of the forecasting ensemble promised in the
project plan: a *global* sequence-to-one LSTM with learned SKU / store
embeddings and calendar context.

Design
------
input sequence : the 28 consecutive daily lags of the target SKU (seq_1..seq_28)
static inputs  : item id embedding, store id embedding, item mean volume,
                 store scaling factor
context inputs : holiday / promotion flags + Fourier seasonality terms
target         : daily sales expressed as a *ratio* of the SKU's own mean volume
                 (stabilises training across series that differ by 2 orders of
                 magnitude)

The model exposes the same `.predict(X)` interface as the scikit-learn models,
so `models_forecast.rollout_single_model` can drive it recursively: at every
forecast step the sequence is rebuilt from the model's own previous outputs.

Usage:
    python src/models_dl.py            # train + quick hold-out report
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from features import FEATURE_COLS_CAL, SEQ_COLS  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(2)

# de-duplicated: item_mean_all already lives inside FEATURE_COLS_CAL
DL_FEATURE_COLS = list(dict.fromkeys(SEQ_COLS + FEATURE_COLS_CAL))


class _LSTMNet(nn.Module):
    def __init__(self, n_cal: int, n_items: int, n_stores: int, hidden: int = 48,
                 emb_items: int = 8, emb_stores: int = 4):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb_items)
        self.store_emb = nn.Embedding(n_stores + 1, emb_stores)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        ctx = hidden + emb_items + emb_stores + n_cal
        self.head = nn.Sequential(
            nn.Linear(ctx, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, seq, cal, item, store):
        out, _ = self.lstm(seq)
        h = out[:, -1, :]
        x = torch.cat([h, self.item_emb(item), self.store_emb(store), cal], dim=1)
        return self.head(x).squeeze(-1)


class LSTMForecaster:
    """Global LSTM with a scikit-learn style fit/predict API."""

    def __init__(self, feature_cols=None, hidden=48, epochs=6, lr=1.5e-3,
                 batch_size=2048, n_train=140_000, seed=config.RANDOM_STATE):
        self.feature_cols = feature_cols or DL_FEATURE_COLS
        self.seq_cols = SEQ_COLS
        self.cal_cols = [c for c in self.feature_cols
                         if c not in self.seq_cols and c not in ("item", "store", "item_mean_all")]
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        self.batch_size, self.n_train, self.seed = batch_size, n_train, seed
        self.net: _LSTMNet | None = None
        self.item_n = self.store_n = 1

    # ---------- tensor assembly ----------
    def _tensors(self, X: pd.DataFrame):
        seq = np.nan_to_num(X[self.seq_cols].to_numpy(dtype=np.float32), nan=0.0)
        scale = np.nan_to_num(X["item_mean_all"].to_numpy(dtype=np.float32), nan=1.0)
        scale = np.where(scale <= 0, 1.0, scale).astype(np.float32)
        ratio = (seq / scale[:, None]).astype(np.float32)
        cal = np.nan_to_num(X[self.cal_cols].to_numpy(dtype=np.float32), nan=0.0).astype(np.float32)
        item = np.clip(X["item"].to_numpy(dtype=np.float32), 1, self.item_n).astype(np.int64)
        store = np.clip(X["store"].to_numpy(dtype=np.float32), 1, self.store_n).astype(np.int64)
        return seq, ratio, cal, item, store, scale

    # ---------- training ----------
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.item_n = int(X["item"].max())
        self.store_n = int(X["store"].max())

        n = min(self.n_train, len(X))
        pos = rng.choice(len(X), size=n, replace=False)
        Xs, ys = X.iloc[pos], y.iloc[pos]
        _, ratio, cal, item, store, scale = self._tensors(Xs)
        target = np.nan_to_num(ys.to_numpy(dtype=np.float32) / scale, nan=0.0)

        seq_t = torch.from_numpy(ratio).unsqueeze(-1)
        cal_t = torch.from_numpy(cal)
        item_t = torch.from_numpy(item)
        store_t = torch.from_numpy(store)
        y_t = torch.from_numpy(target)

        self.net = _LSTMNet(n_cal=len(self.cal_cols), n_items=self.item_n, n_stores=self.store_n,
                            hidden=self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        n_batches = int(np.ceil(n / self.batch_size))
        idx = np.arange(n)
        for ep in range(1, self.epochs + 1):
            rng.shuffle(idx)
            running = 0.0
            t0 = time.time()
            self.net.train()
            for b in range(n_batches):
                sl = idx[b * self.batch_size:(b + 1) * self.batch_size]
                opt.zero_grad()
                pred = self.net(seq_t[sl], cal_t[sl], item_t[sl], store_t[sl])
                loss = loss_fn(pred, y_t[sl])
                loss.backward()
                opt.step()
                running += float(loss.item()) * len(sl)
            if verbose:
                print(f"      epoch {ep}/{self.epochs}  RMSE(ratio)="
                      f"{np.sqrt(running / n):.4f}  ({time.time() - t0:.0f}s)")
        self.net.eval()
        return self

    # ---------- inference ----------
    @torch.no_grad()
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("LSTMForecaster must be fitted before predict()")
        _, ratio, cal, item, store, scale = self._tensors(X)
        out = []
        for i in range(0, len(X), 8192):
            p = self.net(torch.from_numpy(ratio[i:i + 8192]).unsqueeze(-1),
                         torch.from_numpy(cal[i:i + 8192]),
                         torch.from_numpy(item[i:i + 8192]),
                         torch.from_numpy(store[i:i + 8192]))
            out.append(p.numpy())
        pred = np.concatenate(out) * scale
        return np.clip(pred, 0, None)


# --------------------------------------------------------------------------
def run_report() -> dict:
    """Train on the project training window and report a hold-out score."""
    print("=" * 78)
    print("MODULE M3b - LSTM DEEP-LEARNING FORECASTER")
    print("=" * 78)
    ft = pd.read_csv(config.PROCESSED_DIR / "store_demand_features.csv.gz")
    ft["date"] = pd.to_datetime(ft["date"])
    tr = ft[ft["date"] < "2017-07-01"].dropna(subset=DL_FEATURE_COLS)
    va = ft[(ft["date"] >= "2017-04-02") & (ft["date"] < "2017-07-01")].dropna(subset=DL_FEATURE_COLS)
    print(f"    train windows {len(tr):,} | validation windows {len(va):,}")

    t0 = time.time()
    mdl = LSTMForecaster().fit(tr[DL_FEATURE_COLS], tr["sales"])
    p = mdl.predict(va[DL_FEATURE_COLS])
    err = va["sales"].to_numpy(float) - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mask = va["sales"].to_numpy(float) > 1e-6
    mape_val = float(np.mean(np.abs(err[mask] / va["sales"].to_numpy(float)[mask])) * 100)
    rep = {"train_windows": int(len(tr)), "params": int(sum(p.numel() for p in mdl.net.parameters())),
           "one_step_MAE": round(mae, 4), "one_step_RMSE": round(rmse, 4),
           "one_step_MAPE": round(mape_val, 3), "train_sec": round(time.time() - t0, 1)}
    Path(config.METRICS_DIR / "lstm_holdout.json").write_text(json.dumps(rep, indent=2))
    print(f"    one-step hold-out: MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape_val:.2f}%  "
          f"({rep['train_sec']}s, {rep['params']:,} params)")
    return rep


if __name__ == "__main__":
    run_report()
