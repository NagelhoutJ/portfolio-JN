from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any

import torch
import torch.nn as nn
from ray import train

from .data import DataConfig, make_cifar10_loaders
from .model import ModelConfig, CifarCNN


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Mac MPS optioneel: if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def train_tune(config: Dict[str, Any]) -> None:
    """
    Ray Tune entrypoint. 'config' bevat zowel training als model knobs.
    """
    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_cfg = DataConfig(
        data_dir=config.get("data_dir", "data"),
        batch_size=int(config.get("batch_size", 128)),
        num_workers=int(config.get("num_workers", 2)),
    )
    train_loader, val_loader = make_cifar10_loaders(data_cfg)

    model_cfg = ModelConfig(
        num_blocks=int(config["num_blocks"]),
        base_channels=int(config["base_channels"]),
        kernel_size=int(config["kernel_size"]),
        use_batchnorm=bool(config["use_batchnorm"]),
        dropout=float(config["dropout"]),
        mlp_hidden=int(config["mlp_hidden"]),
        num_classes=10,
    )

    device = get_device()
    model = CifarCNN(model_cfg).to(device)

    lr = float(config["lr"])
    weight_decay = float(config.get("weight_decay", 1e-4))
    opt_name = str(config.get("optimizer", "adam")).lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    epochs = int(config.get("epochs", 8))
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

        train_loss = train_loss_sum / max(train_n, 1)

        # ---- validate ----
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                bs = x.size(0)
                val_loss_sum += loss.item() * bs
                val_n += bs
                correct += (logits.argmax(dim=1) == y).sum().item()

        val_loss = val_loss_sum / max(val_n, 1)
        val_acc = correct / max(val_n, 1)

        # Rapporteer aan Ray Train / Tune
        train.report({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })