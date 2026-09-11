import inspect
from typing import Optional

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.bn(self.conv(combined)))
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: Optional[int] = None,
        reduction: int = 16,
        kernel_size: int = 7
    ) -> None:
        super().__init__()
        channels = c1 if c2 is None else c2
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


def register_cbam_to_ultralytics() -> None:
    try:
        import ultralytics.nn.modules as modules
        import ultralytics.nn.tasks as tasks

        setattr(modules, "CBAM", CBAM)
        setattr(tasks, "CBAM", CBAM)

        if not hasattr(tasks, "_cbam_patched"):
            source = inspect.getsource(tasks.parse_model)
            target = "elif m is torch.nn.BatchNorm2d:"
            replacement = """elif m is CBAM or (isinstance(m, str) and m == "CBAM"):
            c2 = ch[f]
            args = [ch[f]]
        elif m is torch.nn.BatchNorm2d:"""

            new_source = source.replace(target, replacement)
            exec_globals = tasks.__dict__.copy()
            exec_globals["CBAM"] = CBAM
            exec(new_source, exec_globals)
            tasks.parse_model = exec_globals["parse_model"]
            tasks._cbam_patched = True
    except Exception as e:
        print(f"Cảnh báo khi tích hợp CBAM vào Ultralytics: {e}")
