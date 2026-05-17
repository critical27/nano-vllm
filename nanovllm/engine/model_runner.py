import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        # 子进程循环等待主进程任务
        while True:
            method_name, args = self.read_shm()  # 从共享内存读取任务
            self.call(method_name, *args)  # 执行任务
            if method_name == "exit":
                break  # 收到退出命令则退出循环

    def read_shm(self):
        # 从共享内存读取任务（子进程用）
        assert self.world_size > 1 and self.rank
        self.event.wait()  # 等待事件触发
        n = int.from_bytes(self.shm.buf[0:4], "little")  # 读取数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化任务
        self.event.clear()  # 清除事件
        return method_name, args

    def write_shm(self, method_name, *args):
        # 向共享内存写入任务（主进程用）
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])  # 序列化任务
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # 写入长度
        self.shm.buf[4:n+4] = data  # 写入数据
        for event in self.event:
            event.set()  # 通知所有子进程

    def call(self, method_name, *args):
        # 调用指定方法
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)  # 主进程写任务到共享内存
        method = getattr(self, method_name, None)  # 获取方法
        return method(*args)  # 执行方法

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # doodle: 计算每张显卡上 KVCache中每个block需要多少字节（需要这么大空间 才能保存token的KV）
        # Transformer中，Q有多少个头，K和V就也有多少个头，满足1对1的关系。
        # 而目前很多模型的做法都采用GQA，即Q和K/V变成了多对一的关系，让多个Q头共享同一个K/V头，这样可以在不显著影响性能的前提下，减少KV缓存的显存占用。
        # 2 * 层数 * 每个kvcache block能装多少 token * 每张卡上KV head数量 * 每个head负责的维度大小 * 数据类型字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

        # 计算可用的kvcache中block数量（考虑显存利用率、已用、峰值、当前分配等）
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        # 分配KV缓存张量，形状为[2, 层数, block数量, block大小, kv head数量, head大小]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        # 遍历模型的所有子模块，将分配好的KV缓存绑定到每一层
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]  # 绑定K缓存
                module.v_cache = self.kv_cache[1, layer_id]  # 绑定V缓存
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        prepare_prefill 把输入的多个序列合并为一个大的一维序列
        cu_seqlens_q/k 告诉注意力内核每条序列的边界
        slot_mapping 用于告诉模型这一批新算出来的K/V要写进 KV cache 的哪个槽位。
        """

        input_ids = []
        positions = []

        # doodle:
        # cu_seqlens 是 cumulative sequence lengths 的缩写，FlashAttention 需要用它来区分批次里每条序列在哪里开始和结束。
        # 序列A：总长 10，已缓存 8，本轮新算 2
        # 序列B：总长 6，已缓存 0，本轮新算 6
        # cu_seqlens_q = [0, 2, 8]  # query 的前缀和，表示每条序列的 query 部分在输入张量中的起始位置
        # cu_seqlens_k = [0, 10, 16] # key 的前缀和，表示每条序列的 key 部分在输入张量中的起始位置
        # 序列A：q 在拼接张量里是 [0,2)，k 是 [0,10)
        # 序列B：q 在拼接张量里是 [2,8)，k 是 [10,16)
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue

            # doodle: 一个block里有多个slot

            # 第一个token在kvcache中的block序号
            start_block = start // self.block_size
            # 最后一个token在kvcache中的block序号（如果正好对齐block边界，则不占用新block）
            end_block = (end + self.block_size - 1) // self.block_size
            # 逐个 block 计算这个 block 里要写入的 slot 范围
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        # 如果有 prefix cache（key 比 query 多），需要准备 block_tables
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        # 转为 CUDA 张量并固定内存，提升推理效率
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 设置推理上下文，包括 cu_seqlens、slot_mapping、block_tables 等
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        和prefill有些相似 区别在于decode阶段只需要输入每个序列的最后一个token和其位置即可
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # prefill 阶段形状变化大、上下文长度差异大，用 eager 最灵活。
            # 手动强制 eager 时，跳过 CUDA Graph。
            # cudagraph 最多只支持 512 的批次大小，因此超过时也使用 eager。
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # cudagraph 只能处理固定形状的输入 所以根据当前批次大小选择对应的 graph 来执行
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]

            # 把当前批次的输入数据和上下文信息写入 graph 变量
            # 本质上是给 cuda graph 的固定缓冲区赋值
            # 后续 graph.replay() 时模型计算会直接使用这些缓冲区里的数据。
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # 回放 cuda graph 来执行模型计算
            # 输出保存在 graph_vars["outputs"] 的前 bs 行里
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # doodle:
        # 支持的 batch_size 列表，需要为每个支持的 batch_size 生成一个对应的 cuda graph
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从大到小录，先捕获最大 bs（512），使内存池以最大规模初始化；后续小 bs 的图和它共享池，小 bs 占的内存一定 <= 大 bs，不会越界。
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])

            # eager 运行一次模型 目的是触发 PyTorch 内部的懒初始化（kernel cache、cublas workspace等），确保后续录制时不会夹杂初始化 kernel。
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            # 在这个上下文里执行的所有 GPU 操作都被"录制"进 graph，而不是实际执行。
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 记录下来这些输入输出 tensor (本质上更像记住了其地址)
        # 后续回放时 模型会直接使用这些地址传递输入输出 避免了每次推理时的内存分配和释放开销。
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
