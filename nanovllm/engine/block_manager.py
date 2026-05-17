from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id    # 物理 KV cache block 的 id
        self.ref_count = 0          # 引用计数 当前有多少 sequence 在使用这个 block
        self.hash = -1              # 这个 block 存的 token_ids 的 hash 用于快速比较两个 block 是否相同
        self.token_ids = []         # 这个 block 存的 token_ids 用于在 hash 相同但不确定是否相同的情况下进行最终比较

    def update(self, hash: int, token_ids: list[int]):
        # hash_blocks中会调用这个函数来更新 block 的 hash 和 token_ids
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


"""
一个 sequence 的 token 被切成逻辑 block：
tokens:      [0 ... 255] [256 ... 511] [512 ...]
logic block:     0            1            2

BlockManager 负责把这些逻辑 block 映射到物理 KV cache block：
seq.block_table = [7, 3, 12]
逻辑 block 0 -> 物理 KV block 7
逻辑 block 1 -> 物理 KV block 3
逻辑 block 2 -> 物理 KV block 12

后面的 ModelRunner.prepare_prefill/prepare_decode 会根据 block_table 生成 slot_mapping 和 block_tables，告诉 attention kernel 把新算出的 K/V 写到哪个 slot。
"""
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # prefix block hash -> 物理 block id
        self.hash_to_block_id: dict[int, int] = dict()

        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 计算一个block的hash值 用于快速比较两个block是否相同
        # hash值是基于token_ids和前一个block的hash值计算的 相当于链式hash
        # 这样可以把一个sequence的所有block的hash值串联起来 只要有一个block不同 整个sequence的hash值就会不同
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        # 注意它没有清空 hash 和 token_ids，也没有删除 hash_to_block_id。
        # 这是有意的：释放后的 block 虽然不再被 running sequence 引用，但它里面的 KV cache 内容还在。
        # 如果之后有相同 prefix 的请求，可以把它从 free list 中“捞回来”复用。
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """
        这个方法用于 prefill 前判断：
        当前 sequence 能复用多少个 prefix blocks？
        剩余 block 是否够分配？
        如果够，返回 num_cached_blocks。
        如果不够，返回 -1。
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        # 不缓存最后一个 block。原因是最后一个 block 可能没满。
        # 最后一个 partial block 后续还会追加 token，不适合当作共享 prefix block 复用。
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # 需要新分配block
                break

            # block可以复用
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1

        # KV cache里剩余的 free block 不够分配了
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1

        # 处理可复用的prefix blocks
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)

        # 分配剩余的blocks
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())

        # 告诉调度器：前面这些 token 的 K/V 已经在 cache 里了，prefill 时可以跳过。
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # 用于decode阶段 每次给sequence追加1个token。
        if len(seq) % self.block_size == 1:
            return len(self.free_block_ids) >= 1
        return True

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """
        用来把刚刚完整算完的 block 登记进 prefix cache。
        它只 hash “本轮新完成的完整 block”。
        """
        # start 是之前已经缓存完成到哪个block
        start = seq.num_cached_tokens // self.block_size
        # end 是本轮执行后，完整覆盖到哪个block
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return

        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        # 更新从start到end的block的hash和token_ids，并把它们登记到hash_to_block_id里
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
