import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出的最后一维是输入的一半
        # 参见Qwen3MLP里是如何保证输入输出维度关系的
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
