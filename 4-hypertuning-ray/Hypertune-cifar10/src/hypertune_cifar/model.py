from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    num_blocks: int = 3          # 2..4
    base_channels: int = 32      # 16..64
    kernel_size: int = 3         # 3 of 5
    use_batchnorm: bool = True
    dropout: float = 0.2         # 0.0..0.5
    mlp_hidden: int = 256        # 128..512
    num_classes: int = 10


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, use_bn: bool, dropout: float):
        super().__init__()
        pad = k // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=not use_bn),
            nn.ReLU(inplace=True),
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_ch))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CifarCNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert 2 <= cfg.num_blocks <= 5
        assert cfg.kernel_size in (3, 5)

        blocks: list[nn.Module] = []
        in_ch = 3
        ch = cfg.base_channels
        for _ in range(cfg.num_blocks):
            blocks.append(ConvBlock(in_ch, ch, cfg.kernel_size, cfg.use_batchnorm, cfg.dropout))
            in_ch = ch
            ch *= 2

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, cfg.mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(cfg.mlp_hidden, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
