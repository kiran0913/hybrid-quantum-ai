"""
quantum-service/main.py

Hybrid Quantum-AI service with:
- Train (XOR / CSV)
- Training curves
- N-feature predict
- Save/Load model persistence
- Anomaly detection semantics:
  anomaly_score = probability
  is_anomaly = anomaly_score >= threshold

Realistic upgrades:
- Confusion matrix + precision/recall/F1 on test split
- Threshold sweep (precision/recall/F1 vs threshold) + best threshold
- Demo dataset endpoint (easy/hard)
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

# -------------------------
# Constants (storage)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
STATE_PATH = os.path.join(MODEL_DIR, "state.pt")
META_PATH = os.path.join(MODEL_DIR, "meta.json")


# -------------------------
# API models
# -------------------------
class TrainRequest(BaseModel):
    epochs: int = Field(default=80, ge=1, le=500)
    lr: float = Field(default=0.06, gt=0.0, le=1.0)
    seed: int = Field(default=42, ge=0, le=10000)
    n_samples: int = Field(default=200, ge=20, le=5000)


class TrainingHistory(BaseModel):
    loss: List[float]
    train_acc: List[float]
    test_acc: List[float]


class ConfusionMatrix(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int


class EvalMetrics(BaseModel):
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: ConfusionMatrix


class ThresholdSweep(BaseModel):
    thresholds: List[float]
    accuracy: List[float]
    precision: List[float]
    recall: List[float]
    f1: List[float]
    best_f1_threshold: float
    best_f1: float


class TrainResponse(BaseModel):
    epochs: int
    final_loss: float
    final_accuracy: float
    test_accuracy: Optional[float] = None
    n_features: Optional[int] = None
    n_samples: Optional[int] = None
    history: Optional[TrainingHistory] = None

    prevalence_test: Optional[float] = None
    eval_at_05: Optional[EvalMetrics] = None
    eval_best_f1: Optional[EvalMetrics] = None
    sweep: Optional[ThresholdSweep] = None


class PredictRequest(BaseModel):
    features: Optional[List[float]] = None
    x1: Optional[float] = None
    x2: Optional[float] = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate(self):
        if self.features is not None:
            if len(self.features) < 1:
                raise ValueError("features must be a non-empty array.")
            return self
        if self.x1 is None or self.x2 is None:
            raise ValueError("Provide either features[] or x1 and x2.")
        return self


class PredictResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold: float


class StatusResponse(BaseModel):
    status: str  # NOT_TRAINED | TRAINING | READY
    trained: bool
    n_features: Optional[int] = None
    persisted: bool = False


class PersistResponse(BaseModel):
    ok: bool
    message: str


# -------------------------
# Core: Hybrid model
# -------------------------
@dataclass(frozen=True)
class QuantumConfig:
    n_qubits: int
    n_layers: int = 2


def make_xor_dataset(n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, 2)).astype(np.float32)
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(np.float32)
    return x, y


class VQCClassifier(nn.Module):
    """Input(n_features) -> angle embedding -> VQC -> expvals(n_qubits) -> linear -> logit."""

    def __init__(self, cfg: QuantumConfig):
        super().__init__()
        self.cfg = cfg
        self.dev = qml.device("default.qubit", wires=cfg.n_qubits)
        weight_shapes = {"weights": (cfg.n_layers, cfg.n_qubits, 3)}

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            qml.AngleEmbedding(inputs * torch.pi, wires=range(cfg.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(cfg.n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(cfg.n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.head = nn.Linear(cfg.n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.qlayer(x)
        return self.head(feats).squeeze(-1)


# -------------------------
# CSV utilities
# -------------------------
def _looks_like_header(row: List[str]) -> bool:
    for cell in row:
        try:
            float(cell)
        except Exception:
            return True
    return False


def _read_csv_bytes(data: bytes) -> Tuple[Optional[List[str]], List[List[str]]]:
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("CSV is empty.")
    header = rows[0] if _looks_like_header(rows[0]) else None
    body = rows[1:] if header else rows
    return header, body


def _select_columns(
    header: Optional[List[str]],
    body: List[List[str]],
    label_col: str,
    feature_cols: str,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cols = max(len(r) for r in body)

    def col_index(token: str) -> int:
        tok = token.strip()
        if tok.lower() == "last":
            return n_cols - 1
        if header and tok in header:
            return header.index(tok)
        if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
            idx = int(tok)
            if idx < 0:
                idx = n_cols + idx
            return idx
        raise ValueError(f"Unknown column: '{token}'. Use name (header) or index (e.g., 0) or 'last'.")

    y_idx = col_index(label_col)

    if feature_cols.strip():
        x_indices = [col_index(t) for t in feature_cols.split(",")]
    else:
        x_indices = [i for i in range(n_cols) if i != y_idx]

    if len(x_indices) < 1:
        raise ValueError("No feature columns selected.")
    if len(x_indices) > 6:
        raise ValueError("Too many features for this demo (max 6).")

    x_rows: List[List[float]] = []
    y_vals: List[float] = []

    for r in body:
        if len(r) < n_cols:
            r = r + [""] * (n_cols - len(r))
        try:
            x_rows.append([float(r[i]) for i in x_indices])
            y_vals.append(float(r[y_idx]))
        except Exception:
            continue

    if not x_rows:
        raise ValueError("No usable numeric rows found. Check CSV columns and values.")

    x = np.array(x_rows, dtype=np.float32)
    y = np.array(y_vals, dtype=np.float32)

    uniq = sorted(set(y.tolist()))
    if len(uniq) > 2:
        raise ValueError(f"Only binary classification supported. Found labels: {uniq[:10]}")
    if len(uniq) == 2:
        y = (y == uniq[1]).astype(np.float32)
    else:
        y = (y > 0).astype(np.float32)

    return x, y


# -------------------------
# Normalization (store + reuse)
# -------------------------
def _fit_normalizer(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_normalizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def _acc_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return float(preds.eq(y).float().mean().item())


# -------------------------
# Metrics
# -------------------------
def _confusion_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrix:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return ConfusionMatrix(tn=tn, fp=fp, fn=fn, tp=tp)


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> EvalMetrics:
    y_pred = (y_prob >= threshold).astype(int)
    cm = _confusion_from_preds(y_true, y_pred)
    total = cm.tn + cm.fp + cm.fn + cm.tp
    accuracy = (cm.tp + cm.tn) / total if total else 0.0
    precision = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) else 0.0
    recall = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return EvalMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion=cm,
    )


def _threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> ThresholdSweep:
    thresholds = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    accs, precs, recs, f1s = [], [], [], []
    best_idx = 0
    best_f1 = -1.0

    for i, t in enumerate(thresholds):
        m = _metrics_at_threshold(y_true, y_prob, float(t))
        accs.append(m.accuracy)
        precs.append(m.precision)
        recs.append(m.recall)
        f1s.append(m.f1)
        if m.f1 > best_f1:
            best_f1 = m.f1
            best_idx = i

    return ThresholdSweep(
        thresholds=[float(x) for x in thresholds.tolist()],
        accuracy=accs,
        precision=precs,
        recall=recs,
        f1=f1s,
        best_f1_threshold=float(thresholds[best_idx]),
        best_f1=float(best_f1),
    )


# -------------------------
# Training
# -------------------------
def _train_eval(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
    n_layers: int = 2,
    test_ratio: float = 0.2,
    record_history: bool = True,
    eval_every: int = 1,
) -> Tuple[VQCClassifier, float, float, float, np.ndarray, np.ndarray, TrainingHistory, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = x.astype(np.float32)
    y = y.astype(np.float32)

    n = len(x)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = idx[:split], idx[split:]

    x_train_np = x[train_idx]
    y_train_np = y[train_idx]
    x_test_np = x[test_idx] if len(test_idx) else x_train_np
    y_test_np = y[test_idx] if len(test_idx) else y_train_np

    mean, std = _fit_normalizer(x_train_np)
    x_train_np = _apply_normalizer(x_train_np, mean, std)
    x_test_np = _apply_normalizer(x_test_np, mean, std)

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    cfg = QuantumConfig(n_qubits=x.shape[1], n_layers=n_layers)
    model = VQCClassifier(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    loss_hist: List[float] = []
    train_acc_hist: List[float] = []
    test_acc_hist: List[float] = []

    final_loss = 0.0
    final_train_acc = 0.0
    final_test_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

        if record_history and (ep % eval_every == 0 or ep == 1 or ep == epochs):
            model.eval()
            with torch.no_grad():
                train_logits = model(x_train)
                train_loss = float(loss_fn(train_logits, y_train).item())
                train_acc = _acc_from_logits(train_logits, y_train)
                test_logits = model(x_test)
                test_acc = _acc_from_logits(test_logits, y_test)

            loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)
            test_acc_hist.append(test_acc)

            final_loss = train_loss
            final_train_acc = train_acc
            final_test_acc = test_acc

    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        test_probs = torch.sigmoid(test_logits).cpu().numpy().astype(np.float32)

    history = TrainingHistory(loss=loss_hist, train_acc=train_acc_hist, test_acc=test_acc_hist)
    return model, final_loss, final_train_acc, final_test_acc, mean, std, history, y_test_np.astype(np.float32), test_probs


# -------------------------
# Persistence helpers
# -------------------------
def _persist_exists() -> bool:
    return os.path.exists(STATE_PATH) and os.path.exists(META_PATH)


def _save_model_to_disk() -> PersistResponse:
    if _MODEL is None or _NORM_MEAN is None or _NORM_STD is None or _N_FEATURES is None:
        return PersistResponse(ok=False, message="No trained model to save.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(_MODEL.state_dict(), STATE_PATH)

    meta = {
        "n_features": int(_N_FEATURES),
        "norm_mean": _NORM_MEAN.tolist(),
        "norm_std": _NORM_STD.tolist(),
        "status": "READY",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return PersistResponse(ok=True, message=f"Saved to {MODEL_DIR}")


def _load_model_from_disk() -> PersistResponse:
    global _MODEL, _STATUS, _NORM_MEAN, _NORM_STD, _N_FEATURES

    if not _persist_exists():
        return PersistResponse(ok=False, message="No saved model found.")

    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        n_features = int(meta["n_features"])
        mean = np.array(meta["norm_mean"], dtype=np.float32)
        std = np.array(meta["norm_std"], dtype=np.float32)

        cfg = QuantumConfig(n_qubits=n_features, n_layers=2)
        model = VQCClassifier(cfg)
        state = torch.load(STATE_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        _MODEL = model
        _N_FEATURES = n_features
        _NORM_MEAN = mean
        _NORM_STD = std
        _STATUS = "READY"
        return PersistResponse(ok=True, message="Loaded saved model.")
    except Exception as exc:
        _MODEL = None
        _N_FEATURES = None
        _NORM_MEAN = None
        _NORM_STD = None
        _STATUS = "NOT_TRAINED"
        return PersistResponse(ok=False, message=f"Failed to load: {exc}")


# -------------------------
# Demo dataset generator (easy/hard)
# -------------------------
def _generate_demo_dataset(rows: int, anomaly_rate: float, seed: int, mode: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = (rng.random(rows) < anomaly_rate).astype(int)

    # failed_logins, bytes_sent, bytes_received, unique_src_ips, hour, session_seconds
    failed_logins = rng.poisson(1.3, rows).astype(float)
    bytes_sent = rng.lognormal(mean=8.7, sigma=0.7, size=rows)
    bytes_recv = rng.lognormal(mean=9.1, sigma=0.8, size=rows)
    unique_ips = (rng.poisson(3.2, rows).astype(float) + 1)
    hour = rng.integers(0, 24, size=rows).astype(float)
    session_seconds = rng.lognormal(mean=6.2, sigma=0.6, size=rows)

    idx = np.where(y == 1)[0]
    if len(idx) > 0:
        if mode == "easy":
            failed_logins[idx] += rng.integers(10, 40, size=len(idx))
            bytes_sent[idx] *= rng.uniform(6, 25, size=len(idx))
            bytes_recv[idx] *= rng.uniform(5, 30, size=len(idx))
            unique_ips[idx] += rng.integers(25, 150, size=len(idx))
            hour[idx] = rng.choice([0, 1, 2, 3, 23], size=len(idx)).astype(float)
            session_seconds[idx] *= rng.uniform(3, 10, size=len(idx))
        else:
            failed_logins[idx] += rng.integers(3, 10, size=len(idx))
            bytes_sent[idx] *= rng.uniform(1.8, 3.0, size=len(idx))
            bytes_recv[idx] *= rng.uniform(1.6, 2.8, size=len(idx))
            unique_ips[idx] += rng.integers(5, 25, size=len(idx))
            night = rng.choice([22, 23, 0, 1, 2, 3, 4], size=len(idx))
            hour[idx] = np.where(rng.random(len(idx)) < 0.6, night, hour[idx]).astype(float)
            session_seconds[idx] *= rng.uniform(1.4, 2.2, size=len(idx))

    data = np.column_stack(
        [
            np.round(failed_logins).astype(int),
            np.round(bytes_sent).astype(int),
            np.round(bytes_recv).astype(int),
            np.round(unique_ips).astype(int),
            np.round(hour).astype(int),
            np.round(session_seconds).astype(int),
            y.astype(int),
        ]
    )
    return data


# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Quantum Service (Hybrid Quantum-AI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL: Optional[VQCClassifier] = None
_STATUS: str = "NOT_TRAINED"
_NORM_MEAN: Optional[np.ndarray] = None
_NORM_STD: Optional[np.ndarray] = None
_N_FEATURES: Optional[int] = None


@app.on_event("startup")
def _startup_load():
    if _persist_exists():
        _load_model_from_disk()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(
        status=_STATUS,
        trained=_MODEL is not None,
        n_features=_N_FEATURES,
        persisted=_persist_exists(),
    )


@app.get("/demo_dataset")
def demo_dataset(
    rows: int = 2000,
    anomaly_rate: float = 0.05,
    seed: int = 7,
    mode: str = "hard",
) -> Response:
    if rows < 50 or rows > 200000:
        raise HTTPException(status_code=400, detail="rows must be 50..200000")
    if not (0.001 <= anomaly_rate <= 0.5):
        raise HTTPException(status_code=400, detail="anomaly_rate must be 0.001..0.5")
    if mode not in ("easy", "hard"):
        raise HTTPException(status_code=400, detail="mode must be 'easy' or 'hard'")

    data = _generate_demo_dataset(rows=rows, anomaly_rate=anomaly_rate, seed=seed, mode=mode)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["failed_logins", "bytes_sent", "bytes_received", "unique_src_ips", "hour", "session_seconds", "is_anomaly"])
    w.writerows(data.tolist())
    payload = out.getvalue().encode("utf-8")

    filename = f"cyber_demo_{mode}_{rows}.csv"
    return Response(
        content=payload,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/save_model", response_model=PersistResponse)
def save_model() -> PersistResponse:
    return _save_model_to_disk()


@app.post("/load_model", response_model=PersistResponse)
def load_model() -> PersistResponse:
    return _load_model_from_disk()


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    global _MODEL, _STATUS, _NORM_MEAN, _NORM_STD, _N_FEATURES
    _STATUS = "TRAINING"

    x, y = make_xor_dataset(req.n_samples, req.seed)
    model, loss, acc, test_acc, mean, std, history, y_test, test_probs = _train_eval(
        x=x,
        y=y,
        epochs=req.epochs,
        lr=req.lr,
        seed=req.seed,
        n_layers=2,
        test_ratio=0.2,
        record_history=True,
        eval_every=1,
    )

    _MODEL = model
    _NORM_MEAN = mean
    _NORM_STD = std
    _N_FEATURES = int(x.shape[1])
    _STATUS = "READY"

    sweep = _threshold_sweep(y_test, test_probs)
    eval_at_05 = _metrics_at_threshold(y_test, test_probs, 0.5)
    eval_best = _metrics_at_threshold(y_test, test_probs, sweep.best_f1_threshold)

    return TrainResponse(
        epochs=req.epochs,
        final_loss=loss,
        final_accuracy=acc,
        test_accuracy=test_acc,
        n_features=int(x.shape[1]),
        n_samples=int(x.shape[0]),
        history=history,
        prevalence_test=float(np.mean(y_test)),
        eval_at_05=eval_at_05,
        eval_best_f1=eval_best,
        sweep=sweep,
    )


@app.post("/train_csv", response_model=TrainResponse)
async def train_csv(
    file: UploadFile = File(...),
    epochs: int = Form(80),
    lr: float = Form(0.06),
    seed: int = Form(42),
    label_col: str = Form("last"),
    feature_cols: str = Form(""),
    test_ratio: float = Form(0.2),
    eval_every: int = Form(1),
) -> TrainResponse:
    global _MODEL, _STATUS, _NORM_MEAN, _NORM_STD, _N_FEATURES
    _STATUS = "TRAINING"

    raw = await file.read()
    try:
        header, body = _read_csv_bytes(raw)
        x, y = _select_columns(header, body, label_col=label_col, feature_cols=feature_cols)
    except Exception as exc:
        _STATUS = "NOT_TRAINED"
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model, loss, acc, test_acc, mean, std, history, y_test, test_probs = _train_eval(
        x=x,
        y=y,
        epochs=int(epochs),
        lr=float(lr),
        seed=int(seed),
        n_layers=2,
        test_ratio=float(test_ratio),
        record_history=True,
        eval_every=max(1, int(eval_every)),
    )

    _MODEL = model
    _NORM_MEAN = mean
    _NORM_STD = std
    _N_FEATURES = int(x.shape[1])
    _STATUS = "READY"

    sweep = _threshold_sweep(y_test, test_probs)
    eval_at_05 = _metrics_at_threshold(y_test, test_probs, 0.5)
    eval_best = _metrics_at_threshold(y_test, test_probs, sweep.best_f1_threshold)

    return TrainResponse(
        epochs=int(epochs),
        final_loss=float(loss),
        final_accuracy=float(acc),
        test_accuracy=float(test_acc),
        n_features=int(x.shape[1]),
        n_samples=int(x.shape[0]),
        history=history,
        prevalence_test=float(np.mean(y_test)),
        eval_at_05=eval_at_05,
        eval_best_f1=eval_best,
        sweep=sweep,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _MODEL is None or _NORM_MEAN is None or _NORM_STD is None or _N_FEATURES is None:
        raise HTTPException(status_code=409, detail="Model not trained yet. Train or Load model first.")

    feats = req.features if req.features is not None else [float(req.x1), float(req.x2)]
    if len(feats) != int(_N_FEATURES):
        raise HTTPException(status_code=400, detail=f"Expected {_N_FEATURES} features but got {len(feats)}.")

    threshold = float(req.threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1.")

    x_np = np.array([feats], dtype=np.float32)
    x_np = _apply_normalizer(x_np, _NORM_MEAN, _NORM_STD)
    x_t = torch.tensor(x_np, dtype=torch.float32)

    _MODEL.eval()
    with torch.no_grad():
        logit = _MODEL(x_t)
        score = float(torch.sigmoid(logit).item())

    return PredictResponse(
        anomaly_score=score,
        is_anomaly=bool(score >= threshold),
        threshold=threshold,
    )
