import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 并行计算Q、K、V的线性层
        # 每张卡负责一部分head的Q、K、V计算
        qkv = self.qkv_proj(hidden_states)

        # 如果只有一张卡
        # Q 大小为 [B, T, num_heads * head_dim]
        # K 大小为 [B, T, num_kv_heads * head_dim]
        # V 大小为 [B, T, num_kv_heads * head_dim]

        # 如果有多张卡
        # 每张卡的 Q 大小为 [B, T, num_heads * head_dim // tp_size]
        # 每张卡的 K 大小为 [B, T, num_kv_heads * head_dim // tp_size]
        # 每张卡的 V 大小为 [B, T, num_kv_heads * head_dim // tp_size]
        # 切分的是输出维度 多张卡之间不需要通信 使用的是ColumnParallelLinear
        # 即输入是完整的 hidden_states 每张卡都算完整输入对应的一部分输出 多张卡之间不需要通信
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE(Rotary Position Embedding) 注入位置信息
        # 大小不变
        q, k = self.rotary_emb(positions, q, k)

        # 每张卡计算attention输出的一部分
        # 每张卡的输出大小为 [B, T, num_heads * head_dim // tp_size]
        o = self.attn(q, k, v)

        # o_proj 用的是 RowParallelLinear
        # 合并多张卡的attention输出
        # 对于所有卡来说整体完成的是 [B, T, num_heads * head_dim] -> [B, T, hidden_size]
        # 对于每张卡来说输入是 [B, T, num_heads * head_dim // tp_size] 输出是 [B, T, hidden_size // tp_size]
        # 需要通信把每张卡的输出加起来得到完整的输出 (对应代码中的all_reduce)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # x: [B, T, C]
        # C就是hidden_size

        # gate_up_proj 用的是 ColumnParallelLinear
        # 对于所有卡来说是整体完成的是 [B, T, C] -> [B, T, 2 * intermediate_size]
        # 而每张卡的输出是 [B, T, 2 * intermediate_size // tp_size]
        # 多张卡之间不需要互相通信
        gate_up = self.gate_up_proj(x)

        # 对于所有卡来说整体完成的是 [B, T, 2 * intermediate_size] -> [B, T, intermediate_size]
        # 而每张卡的输出是 [B, T, intermediate_size // tp_size]
        # act = silu(gate) * up 是逐元素的操作 多张卡之间不需要通信
        x = self.act_fn(gate_up)

        # down_proj 用的是 RowParallelLinear
        # 对于所有卡来说整体完成的是 [B, T, intermediate_size] -> [B, T, C]
        # 对于每张卡来说输入是 [B, T, intermediate_size // tp_size] 输出是 [B, T, C // tp_size]
        # 需要通信把每张卡的输出加起来得到完整的输出 即代码中的all_reduce(y)
        # rank0: y0 = x0 @ w0^T + b0    大小为 [B, T, C // tp_size]
        # rank1: y1 = x1 @ w1^T + b1    大小为 [B, T, C // tp_size]
        # ...
        # y = y0 + y1 + ...             大小为 [B, T, C]
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,        # hidden_states是输入x
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        本质上是为了了实现类似下面的计算 为了省中间变量 把residual来回传递
        u = x + Attention(RMSNorm(x))
        y = u + MLP(RMSNorm(u))
        """

        if residual is None:
            # residual = x
            # hidden_states = RMSNorm(x)
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # residual = residual + x
            # hidden_states = RMSNorm(residual + x)
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # hidden_states = Attention(RMSNorm(x))
        hidden_states = self.self_attn(positions, hidden_states)

        # u = x + Attention(RMSNorm(x))
        # hidden_states = RMSNorm(u)
        # residual = u
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # hidden_states = MLP(RMSNorm(u))
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        整体模型的前向计算流程如下：
        1. 首先通过词嵌入层将输入的 token 转换为 embedding
        2. 经过多层 Qwen3DecoderLayer 每层中包含两个子层
           * Attention子层 Qwen3Attention: x = x + Attention(RMSNorm(x))
           * MLP子层 Qwen3MLP: x = x + MLP(RMSNorm(x))
        3. 最后再进行一次 RMSNorm 得到最终输出
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
