import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        # RMSNorm 只学习缩放参数 gamma，不像 LayerNorm 那样额外学习 bias。
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        RMS(x) = sqrt(mean(x^2) + eps)
        RMSNorm(x) = x / RMS(x) * weight
        """
        orig_dtype = x.dtype
        x = x.float()

        var = x.pow(2).mean(dim=-1, keepdim=True)
        # x / sqrt(E[x^2] + eps)
        x.mul_(torch.rsqrt(var + self.eps))

        # 转回原 dtype，并乘以可学习的逐通道缩放参数。
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x = x + residual
        residual = x
        x = RMSNorm(x)
        return x, residual
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
