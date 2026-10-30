"""Small mask-aware learner and training-only pseudo-label selectors (pilot013)."""

import copy
import hashlib
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .evidence_profile_demo import BLOCKS
from .temporal_comparison import RecurrentGesture
from .boundary_train import state_hash

FEATURES = 149
PARAMETERS = 49840
MASKS = np.array([[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)


class Model(RecurrentGesture):
    def __init__(self):
        super().__init__(bidirectional=False)
        self.recurrent = nn.LSTM(FEATURES, 47, batch_first=True)
        self.head[2] = nn.Linear(47 * 4, 64)
        if sum(p.numel() for p in self.parameters()) != PARAMETERS:
            raise ValueError("Unexpected mask-aware model size.")

    def embedding(self, x):
        sequence, _ = self.recurrent(x)
        return self.head[:4](sequence.transpose(1, 2))


def observed(x, mask=0):
    """Copy inputs; availability is distinct from a physically zero channel."""
    values = torch.as_tensor(x, dtype=torch.float32)
    codes = torch.as_tensor(mask, dtype=torch.long)
    if codes.ndim == 0:
        codes = codes.repeat(len(values))
    if values.ndim != 3 or values.shape[1:] != (32, 146) or codes.shape != (len(values),) or torch.any((codes < 0) | (codes > 3)):
        raise ValueError("Require32x146 clips and one of four availability patterns.")
    result = torch.cat((values.clone(), torch.ones((len(values), 32, 3))), dim=2)
    availability = torch.from_numpy(MASKS)[codes]
    for j, name in enumerate(("pose", "articulation", "transport")):
        result[:, :, list(BLOCKS[name])] *= availability[:, j, None, None]
    result[:, :, 146:] = availability[:, None, :]
    return result


@dataclass(frozen=True)
class LearnerPool:
    x: np.ndarray
    labeled: np.ndarray
    y: np.ndarray
    teacher_a: np.ndarray
    teacher_b: np.ndarray
    unlabeled: np.ndarray

    @classmethod
    def load(cls, path):
        keys = {"x", "labeled", "y", "teacher_a", "teacher_b", "unlabeled"}
        with np.load(path, allow_pickle=False) as a:
            if set(a.files) != keys:
                raise ValueError("Foreign/hidden-label metadata in learner bundle.")
            result = cls(**{k: a[k].copy() for k in keys})
        n = len(result.x)
        if result.x.shape != (n, 32, 146) or not np.isfinite(result.x).all():
            raise ValueError("Invalid learner inputs.")
        if len(result.labeled) != len(result.y) or not np.array_equal(np.sort(np.r_[result.labeled, result.unlabeled]), np.arange(n)):
            raise ValueError("Training indices must be a disjoint complete partition.")
        if set(result.teacher_a) & set(result.teacher_b) or set(np.r_[result.teacher_a, result.teacher_b]) != set(result.labeled):
            raise ValueError("Teacher label groups must be disjoint and exhaust labeled inputs.")
        if set(result.y) != set(range(8)):
            raise ValueError("All eight commands required.")
        return result

    def labels_for(self, indices):
        labels = dict(zip(self.labeled.tolist(), self.y.tolist()))
        if any(int(i) not in labels for i in indices):
            raise ValueError("A hidden training label was requested.")
        return np.array([labels[int(i)] for i in indices], dtype=np.int64)


def infer(model, x, views=False):
    model.eval()
    with torch.inference_mode():
        logits = torch.stack([model(observed(x, m)) for m in (range(4) if views else (0,))]).numpy()
        embedding = model.embedding(observed(x)).numpy()
    return logits, embedding


def probabilities(logits):
    values = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=-1, keepdims=True)


def responses(logits):
    if logits.ndim != 3 or logits.shape[0] != 4 or logits.shape[2] != 8:
        raise ValueError("Require four views of eight-class logits.")
    odds = np.empty_like(logits, dtype=np.float64)
    for c in range(8):
        other = np.delete(logits, c, axis=2).astype(np.float64)
        mx = other.max(axis=2)
        rest = mx + np.log(np.exp(other - mx[:, :, None]).sum(axis=2))
        odds[:, :, c] = logits[:, :, c] - rest
    return (odds[0][None] - odds[1:]).transpose(1, 2, 0)


def support(query, bank, anchor_y, anchor_p, proposed, config):
    """No unknown query labels are accepted by this interface."""
    center = np.median(bank[np.arange(len(bank)), anchor_y], axis=0)
    scale = np.maximum(config["scale_floor"], 1.4826 * np.median(np.abs(bank[np.arange(len(bank)), anchor_y] - center), axis=0))
    weights = np.zeros(len(query), dtype=np.float64)
    details = []
    for c in range(8):
        anchors = np.flatnonzero((anchor_y == c) & (anchor_p.argmax(1) == c) & (anchor_p[:, c] >= config["anchor_confidence"]))
        info = {"class": c, "eligible_anchors": len(anchors), "supported": len(anchors) >= config["minimum_anchors"]}
        if info["supported"]:
            b = bank[anchors, c] / scale
            distances = ((b[:, None] - b[None]) ** 2).sum(2)
            np.fill_diagonal(distances, np.inf)
            radius = max(config["radius_floor_squared"], float(np.median(distances.min(1))))
            indices = np.flatnonzero(proposed == c)
            d = ((query[indices, c, None, :] / scale - b[None]) ** 2).sum(2).min(1) if len(indices) else np.array([])
            weights[indices] = np.where(d <= radius, np.exp(-d / radius), 0)
            info["radius_squared"] = radius
        details.append(info)
    return weights, details


def select(teacher_a, teacher_b, anchors_a, anchors_b, anchor_y_a, anchor_y_b, config, embedding=False):
    pa, pb = (probabilities(t[0][0]) for t in (teacher_a, teacher_b))
    proposed = pa.argmax(1)
    base = (proposed == pb.argmax(1)) & (pa.max(1) >= config["confidence"]) & (pb.max(1) >= config["confidence"])
    q = (pa + pb) / 2
    info = []
    weights = base.astype(np.float64)
    for candidate, anchor, y in ((teacher_a, anchors_a, anchor_y_a), (teacher_b, anchors_b, anchor_y_b)):
        anchor_p = probabilities(anchor[0][0])
        if embedding:
            query = np.repeat(candidate[1][:, None, :], 8, axis=1)
            bank = np.repeat(anchor[1][:, None, :], 8, axis=1)
        else:
            query, bank = responses(candidate[0]), responses(anchor[0])
        w, details = support(query, bank, y, anchor_p, proposed, config)
        weights = np.minimum(weights, w)
        info.append(details)
    return q.astype(np.float32), weights.astype(np.float32), base, info


def matched_weights(profile, base, q, seed, random=False):
    result = np.zeros_like(profile)
    rng = np.random.default_rng(seed + 9_000_000)
    for c in range(8):
        eligible = np.flatnonzero(base & (q.argmax(1) == c))
        values = np.sort(profile[q.argmax(1) == c][profile[q.argmax(1) == c] > 0])[::-1]
        if len(values) > len(eligible):
            raise ValueError("Cannot match admitted class count.")
        chosen = rng.permutation(eligible) if random else eligible[np.lexsort((eligible, -q[eligible].max(1)))]
        result[chosen[:len(values)]] = values
    return result


def fit(x, y, vx, vy, seed, config, subset=False, ux=None, target=None, weight=None, steps=None):
    """Only revealed labels and soft teacher targets enter gradient updates."""
    steps = config["steps"] if steps is None else steps
    interval = steps // config["validation_checks"]
    torch.manual_seed(seed)
    model = Model()
    initial = state_hash(model)
    rng = torch.Generator().manual_seed(seed + 1_000_000)
    masks = torch.Generator().manual_seed(seed + 2_000_000)
    unlabeled_rng = torch.Generator().manual_seed(seed + 3_000_000)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    validation_x, validation_y = observed(vx), torch.from_numpy(vy)
    counts = np.bincount(y, minlength=8)
    if np.any(counts < 5):
        raise ValueError("Need at least five revealed labels per class.")
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (8 * counts), dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    active = ux is not None and weight is not None and np.any(weight > 0)
    if active:
        if target.shape != (len(ux), 8) or weight.shape != (len(ux),) or np.any(weight < 0) or np.any(weight > 1) or not np.isfinite(target).all() or not np.allclose(target.sum(1), 1):
            raise ValueError("Invalid frozen pseudo-targets/weights.")
        ut, qt, wt = torch.from_numpy(ux), torch.from_numpy(target), torch.from_numpy(weight)
    history, best_loss, best, best_step = [], float("inf"), None, None
    batch_hash, mask_hash, u_hash = (hashlib.sha256() for _ in range(3))
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        indices = torch.randint(len(tx), (config["batch_size"],), generator=rng)
        mask = torch.multinomial(torch.tensor([.5, 1/6, 1/6, 1/6]), len(indices), replacement=True, generator=masks)
        if not subset:
            mask.zero_()
        batch_hash.update(indices.numpy().tobytes()); mask_hash.update(mask.numpy().tobytes())
        optimizer.zero_grad()
        loss = loss_fn(model(observed(tx[indices], mask)), ty[indices])
        if active:
            ui = torch.randint(len(ut), (config["batch_size"],), generator=unlabeled_rng)
            u_hash.update(ui.numpy().tobytes())
            soft_ce = -(qt[ui] * model(observed(ut[ui])).log_softmax(1)).sum(1)
            loss = loss + min(1.0, step / config["ramp_steps"]) * (wt[ui] * soft_ce).mean()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss.")
        loss.backward(); optimizer.step()
        if step % interval == 0:
            model.eval()
            with torch.inference_mode():
                val = nn.functional.cross_entropy(model(validation_x), validation_y).item()
            history.append({"step": step, "selection_loss": val})
            if val < best_loss:
                best_loss, best, best_step = val, copy.deepcopy(model.state_dict()), step
    model.load_state_dict(best); model.eval()
    return model, history, {"initial_state_sha256": initial, "batch_indices_sha256": batch_hash.hexdigest(),
        "mask_sha256": mask_hash.hexdigest(), "unlabeled_indices_sha256": u_hash.hexdigest(), "optimizer_steps": steps,
        "validation_checks": len(history), "best_step": best_step, "training_class_counts": counts.tolist(),
        "training_labels": len(y), "active_unsupervised_loss": bool(active), "fit_seconds": time.perf_counter() - started}
