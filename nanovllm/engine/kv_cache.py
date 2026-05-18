import torch
from torch import nn

from nanovllm.engine.sequence import Sequence


class KVCacheLayer:

    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.k = k
        self.v = v


class KVCache:

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
         # 分配KVCache的内存 形状为[2, decoder层数, block数量, block大小, kv head数量, head大小]
        self.tensor = torch.empty(
            2,
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_model_config(cls, config, hf_config, world_size: int, device: torch.device | str = "cuda"):
        num_kv_heads = hf_config.num_key_value_heads // world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        return cls(
            num_layers=hf_config.num_hidden_layers,
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=hf_config.dtype,
            device=device,
        )

    @staticmethod
    def estimate_num_blocks(config, hf_config, world_size: int) -> int:
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # doodle: 计算每张显卡上 KVCache中每个block需要多少字节（需要这么大空间 才能保存token的KV）
        # Transformer中，Q有多少个头，K和V就也有多少个头，满足1对1的关系。
        # 而目前很多模型的做法都采用GQA，即Q和K/V变成了多对一的关系，让多个Q头共享同一个K/V头，这样可以在不显著影响性能的前提下，减少KV缓存的显存占用。
        # 2 * 层数 * 每个kvcache block能装多少 token * 每张卡上KV head数量 * 每个head负责的维度大小 * 数据类型字节数
        block_bytes = 2 * hf_config.num_hidden_layers * config.kvcache_block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

        # 计算可用的kvcache中block数量（考虑显存利用率、已用、峰值、当前分配等）
        return int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes

    @property
    def block_bytes(self):
        return 2 * self.num_layers * self.block_size * self.num_kv_heads * self.head_dim * self.dtype.itemsize

    def layer(self, layer_id: int) -> KVCacheLayer:
        return KVCacheLayer(self.tensor[0, layer_id], self.tensor[1, layer_id])

    def bind_model(self, model: nn.Module):
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "bind_kv_cache"):
                module.bind_kv_cache(self.layer(layer_id))
                layer_id += 1
        assert layer_id == self.num_layers

    # 给定一个逻辑 block id 和这个 block 内的 token offset
    # 计算这个 token 在 KV cache 中对应的 slot id
    def slot(self, block_id: int, offset: int) -> int:
        assert 0 <= offset < self.block_size
        return block_id * self.block_size + offset

    def slots(self, block_id: int, start: int, end: int) -> range:
        assert 0 <= start <= end <= self.block_size
        slot_start = self.slot(block_id, start)
        slot_end = self.slot(block_id, 0) + end
        return range(slot_start, slot_end)

    def build_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def build_slot_mapping_for_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        slot_mapping = []
        for seq in seqs:
            # warmup阶段的序列可能没有block_table 直接跳过
            if not seq.block_table:
                continue
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            # 第一个token在kvcache中的block序号
            start_block = start // self.block_size
            # 最后一个token在kvcache中的block序号（如果正好对齐block边界，则不占用新block）
            end_block = (end + self.block_size - 1) // self.block_size
            # 逐个 block 计算这个 block 里要写入的 slot 范围
            for i in range(start_block, end_block):
                block_id = seq.block_table[i]
                block_start = start % self.block_size if i == start_block else 0
                block_end = end - i * self.block_size if i == end_block - 1 else self.block_size
                slot_mapping.extend(self.slots(block_id, block_start, block_end))
        return torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def build_slot_mapping_for_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        slot_mapping = [self.slot(seq.block_table[-1], seq.last_block_num_tokens - 1) for seq in seqs]
        return torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
