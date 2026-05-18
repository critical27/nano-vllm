import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.engine.kv_cache import KVCacheLayer
from nanovllm.utils.context import get_context


"""
写入KV Cache
"""
@triton.jit
def store_kvcache_kernel(
    key_ptr,            # 当前这一层 attention 新算出来的 K 的指针，大小是 [T, num_kv_heads, head_dim]
    key_stride,         # key[i]和key[i+1]之间的距离 单位是元素个数 不是字节数 即 num_kv_heads * head_dim
    value_ptr,          # 当前这一层 attention 新算出来的 V 的指针，大小是 [T, num_kv_heads, head_dim]
    value_stride,       # value[i]和value[i+1]之间的距离 单位是元素个数 不是字节数 即 num_kv_heads * head_dim
    k_cache_ptr,        # KV Cache 中 K 的指针，大小是 [num_slots, num_kv_heads, head_dim]
    v_cache_ptr,        # KV Cache 中 V 的指针，大小是 [num_slots, num_kv_heads, head_dim]
    slot_mapping_ptr,   # 每个 token 对应的 KV Cache slot id，大小是 [T]
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    # slot = 要写入到KV cache的哪个slot
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return

    # doodle:
    # 虽然传入的key_ptr和value_ptr虽然是多个token的K和V的起始地址(一次attention计算可以得到多个token的K和V 主要是prefill阶段)
    # 但是每个线程只负责写入一个token的K和V到KV cache里对应的slot位置上

    # 这里要写入的K和V 是单个token的K和V 大小是[num_kv_heads * head_dim] (因为使用了GQA 而不是MHA的[num_heads * head_dim])
    # 展开成一维后 大小为 D = n_kv_heads * head_dim
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    # D = num_kv_heads * head_dim
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    # 启动N个Triton线程 每个线程负责写入一个token的K和V到KV cache里对应的slot位置上
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def bind_kv_cache(self, layer: KVCacheLayer):
        self.k_cache = layer.k
        self.v_cache = layer.v

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # context中包含
        # is_prefill      当前是 prefill 还是 decode
        # slot_mapping    新 K/V 应该写到哪些 cache slot
        # cu_seqlens_q    prefill 时 q 的序列边界
        # cu_seqlens_k    prefill 时 k 的序列边界
        # max_seqlen_q
        # max_seqlen_k
        # context_lens    每条序列当前长度
        # block_tables    每条序列的block table
        context = get_context()

        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            # 如果有 prefix cache，说明当前序列的某个prefix的K/V已经在KV Cache里了。
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            # varlen FlashAttention
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # decode直接使用KV Cache里的K/V来计算attention输出
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
