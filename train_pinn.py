"""
Phase 2 (cont'd): Train the PINN on ur5_ik_dataset.npz.

Loads train/val splits, trains IKPinn with the combined physics-informed
loss from pinn_model.py, tracks per-component losses so you can confirm all
four terms are contributing (not just data loss dominating), and saves the
best checkpoint by validation FK-consistency error (the metric that best
reflects "will this be a good warm-start for the classical solver," since
that's the actual downstream use case -- not just matching training labels).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pinn_model import IKPinn, total_loss, fk_consistency_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 60
BATCH_SIZE = 256
LR = 1e-3
CHECKPOINT_PATH = "pinn_ur5_best.pt"


class IKDataset(Dataset):
    def __init__(self, npz_path, split):
        d = np.load(npz_path)
        self.q = torch.tensor(d[f"{split}_q"], dtype=torch.float32)
        self.pos = torch.tensor(d[f"{split}_pos"], dtype=torch.float32)
        self.quat = torch.tensor(d[f"{split}_quat"], dtype=torch.float32)

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return self.pos[idx], self.quat[idx], self.q[idx]


def run_epoch(model, loader, optimizer=None):
    """optimizer=None -> eval mode, no gradient step (used for val)."""
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"data": 0.0, "fk": 0.0, "limits": 0.0, "torque": 0.0, "total": 0.0}
    n_batches = 0

    for pos, quat, q_true in loader:
        pos, quat, q_true = pos.to(DEVICE), quat.to(DEVICE), q_true.to(DEVICE)

        with torch.set_grad_enabled(is_train):
            q_pred = model(pos, quat)
            loss, comps = total_loss(q_pred, q_true, pos, quat)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for k in totals:
            totals[k] += comps[k]
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    train_ds = IKDataset("ur5_ik_dataset.npz", "train")
    val_ds = IKDataset("ur5_ik_dataset.npz", "val")
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = IKPinn().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_fk = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        val_metrics = run_epoch(model, val_loader, optimizer=None)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train total={train_metrics['total']:.4f} "
            f"(data={train_metrics['data']:.4f} fk={train_metrics['fk']:.4f} "
            f"limits={train_metrics['limits']:.5f} torque={train_metrics['torque']:.5f}) | "
            f"val fk={val_metrics['fk']:.4f}"
        )

        if val_metrics["fk"] < best_val_fk:
            best_val_fk = val_metrics["fk"]
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "val_fk": best_val_fk},
                CHECKPOINT_PATH,
            )

    print(f"\nBest val FK-consistency loss: {best_val_fk:.6f}")
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")
