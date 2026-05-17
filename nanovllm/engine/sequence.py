from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256                      # 每个逻辑 block 容纳的 token 数
    counter = count()                     # 全局自增序列 ID 生成器

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)                # 请求唯一 ID
        self.status = SequenceStatus.WAITING                # 调度状态：等待/运行/结束
        self.token_ids = copy(token_ids)                    # 当前完整 token 序列（prompt + completion）
        self.last_token = token_ids[-1]                     # 最近一个 token，decode 阶段直接使用
        self.num_tokens = len(self.token_ids)               # 序列总 token 数
        self.num_prompt_tokens = len(token_ids)             # prompt token 数（初始化后不变）
        self.num_cached_tokens = 0                          # 已写入 KV cache 的 token 数
        self.num_scheduled_tokens = 0                       # 当前调度轮次要处理的 token 数
        self.is_prefill = True                              # 当前是否处于 prefill 阶段
        self.block_table = []                               # 逻辑 block -> 物理 KV block 的映射表
        self.temperature = sampling_params.temperature      # 采样温度
        self.max_tokens = sampling_params.max_tokens        # 最大生成长度（completion）
        self.ignore_eos = sampling_params.ignore_eos        # 是否忽略 eos 提前终止

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        # 已生成 completion 长度（不含 prompt）
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        # prompt 部分 token
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        # completion 部分 token
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # 当前序列占用的逻辑 block 数（向上取整）
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 最后一个逻辑 block 内实际 token 数
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
