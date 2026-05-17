import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        # num_embeddings指的实际是词表大小 即 vocab_size
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0

        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 每个分区只持有词表的一部分 大小为[num_embeddings_per_partition, embedding_dim]
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 只有在输入的token_id在当前分区的词表范围内时才进行embedding 否则置为0
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将token_id映射到当前分区的词表范围内
            # x大小是[B, T]
            x = mask * (x - self.vocab_start_idx)
        # y大小是[B, T, C]
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 每张卡只算自己负责的那部分 embedding，最后用 all_reduce 合成完整结果
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # 假设一个batch里有3个序列，长度分别是[4, 2, 3]。把它们拼平后，x的长度是9。
            # cu_seqlens_q 是前缀和：[0, 4, 6, 9]。
            # last_indices就是每个序列最后一个token的索引：[3, 5, 8]。
            # 通过x[last_indices]就得到了每个序列最后一个token的embedding。

            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        # 通过本卡的词表权重矩阵计算logits
        # 只需要每个序列最后一个token的logits的原因 采样只需要最后一个token的logits就能决定其概率分布了
        # 前面的token的logits仍然会参与attention计算 以及保存到KV cache 但在采样阶段是用不到的
        logits = F.linear(x, self.weight)

        # 每张卡只算自己负责的那部分 logits，最后用 gather 合成完整结果
        if self.tp_size > 1:
            # rank0分配一个全局的logits张量来接收所有卡的logits
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            #每张卡把自己的logits发送到rank0的all_logits里
            dist.gather(logits, all_logits, 0)
            # rank0把所有卡的logits拼起来得到完整的logits 后续采样输出只在rank0进行
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
