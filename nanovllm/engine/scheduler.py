from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        从 waiting 和 running 两个队列中挑出一批要执行的序列，并返回这批序列以及它们属于 prefill 还是 decode 阶段。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 第一次调度这个 sequence
                # 获取这个序列可以使用的缓存 block 数 (prefix caching)
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # 如果当前剩余预算 remaining 不足以覆盖这个序列的num_tokens
            # 只允许第一个被调度的序列做chunked prefill
            # 后面的序列必须能一次性塞下，否则就不调度这个序列
            # 避免多个序列都做 chunked prefill 导致每个序列每轮都只能调度一点点 降低效率
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break

            if not seq.block_table:
                # 分配 KV cache block 给这个序列
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 如果这个序列的所有 token 都被调度了（可能分多轮调度完成），就把它从 waiting 队列移到 running 队列
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # 每次调度一个序列生成一个 token，直到生成完或者达到 max_num_batched_tokens 的限制
            seq = self.running.popleft()
            # 检查是否有足够多的 KV cache block 可以让这个序列继续生成下一个 token
            while not self.block_manager.can_append(seq):
                # 不够的话 会抢占其他正在运行的序列
                # 即将其从 running 队列移回 waiting 队列 并释放它占用的 KV cache block
                if self.running:
                    self.preempt(self.running.pop())
                # 没有别的序列时 只能将自己移回 waiting 队列
                else:
                    self.preempt(seq)
                    break
            else:
                # 调度成功
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # 从 running 队列移回 waiting 队列 并释放它占用的 KV cache block
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            # 对这个序列已填满的block计算哈希，保存到block manager中
            # 供后续请求prefix caching时快速匹配使用
            self.block_manager.hash_blocks(seq)
            # 更新这个序列的 num_cached_tokens 和 num_scheduled_tokens
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # 当前序列仍然处于 chunked prefill 阶段
                continue

            # 将模型输出添加的序列中
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
