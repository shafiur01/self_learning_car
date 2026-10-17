"""Train on recording sessions; select a model using separate validation sessions."""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import CONVENTION, DATA, FRAMES, LABELS, ROOT, SECONDS, VERSION, load_clip, load_dataset
from .model import GestureNet, load_model


def tensors(samples):
    return (torch.from_numpy(np.stack([s[0] for s in samples])),
            torch.tensor([s[1] for s in samples], dtype=torch.long))


def require_classes(samples, split, minimum):
    counts = np.bincount([s[1] for s in samples], minlength=len(LABELS))
    missing = [f"{label}: {count}/{minimum}" for label, count in zip(LABELS, counts) if count < minimum]
    if missing:
        raise ValueError(f"Record more {split} clips before continuing: " + ", ".join(missing))
    return counts


def report(model, samples):
    x, y = tensors(samples)
    with torch.inference_mode():
        predicted = model(x).argmax(dim=1)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for truth, guess in zip(y.tolist(), predicted.tolist()):
        matrix[truth, guess] += 1
    return {"accuracy": float((predicted == y).float().mean()),
            "labels": list(LABELS), "confusion_matrix_actual_rows_predicted_columns": matrix.tolist(),
            "clips": len(samples)}


def train_model(dataset, epochs, seed):
    counts = require_classes(dataset["train"], "train", 5)
    require_classes(dataset["validation"], "validation", 2)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = GestureNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    weights = torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    x, y = tensors(dataset["train"])
    vx, vy = tensors(dataset["validation"])
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(x))
        for indices in permutation.split(32):
            optimizer.zero_grad()
            loss = loss_fn(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = nn.functional.cross_entropy(model(vx), vy).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}: validation loss {validation_loss:.4f}")
    if best_state is None:
        raise ValueError("Training produced no finite validation loss.")
    model.load_state_dict(best_state)
    return model, best_epoch, report(model, dataset["validation"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", type=Path, help="Evaluate a saved model on untouched test clips; do not train")
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.data)
        if args.evaluate:
            require_classes(dataset["test"], "test", 1)
            model, checkpoint = load_model(args.evaluate)
            test_sessions = {load_clip(s[2])[3]["session"] for s in dataset["test"]}
            if test_sessions.intersection(checkpoint["development_sessions"]):
                raise ValueError("Test sessions were already used to develop this model.")
            print(json.dumps(report(model, dataset["test"]), indent=2))
            print("This measures complete clips. Also test continuous video for accidental commands.")
            return
        if args.epochs < 1:
            raise ValueError("Epochs must be at least 1.")
        model, best_epoch, validation = train_model(dataset, args.epochs, args.seed)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = ROOT / "runs" / run_id
        output.mkdir(parents=True)
        files = [s[2] for split in ("train", "validation") for s in dataset[split]]
        sessions = sorted({load_clip(path)[3]["session"] for path in files})
        checkpoint = {
            "state_dict": model.state_dict(), "schema_version": VERSION,
            "convention": CONVENTION, "labels": list(LABELS), "frames": FRAMES,
            "seconds": SECONDS, "seed": args.seed, "best_epoch": best_epoch,
            "development_sessions": sessions, "development_files": files,
        }
        torch.save(checkpoint, output / "model.pt")
        (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(f"Validation accuracy: {validation['accuracy']:.1%} (not a test result)")
        print(f"Saved: {output / 'model.pt'}")
        print("Review validation.json. Keep the old model until this candidate improves your tests.")
        print(f"Try: python -m gesture_commands.app --model {output / 'model.pt'}")
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
