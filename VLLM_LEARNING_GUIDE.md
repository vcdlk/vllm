# vLLM 主干学习与断点调试指南

本文用于从源码理解 vLLM V1。目标不是逐行读完仓库，而是先建立一条
可运行、可观察、可验证的主干，再逐步加入 KV Cache、并行和推测解码等能力。

> 源码持续变化。调试时优先按函数名设置断点，文中的行号只用于快速定位。

## 1. 核心心智模型

不要跟着所有 `if` 分支走，而要跟着一个请求的状态走。

```text
Prompt
  -> Renderer / InputProcessor
  -> Request
  -> Scheduler
  -> KV Cache allocation
  -> ModelExecutor / Worker / ModelRunner
  -> model forward / sampling
  -> Scheduler updates Request
  -> OutputProcessor / detokenizer
  -> RequestOutput
```

Scheduler 最重要的不变量是：

> 每轮尝试让 `request.num_computed_tokens` 追上
> `request.num_tokens_with_spec`，同时受到 token budget 和 KV Cache
> 容量限制。

这套规则统一覆盖：

- Prompt prefill：两者通常相差多个 token。
- 普通 decode：两者通常相差一个 token。
- Chunked prefill：一次只计算 prompt 的一部分。
- Speculative decoding：一次可能安排多个候选 token。

相关说明位于
[`Scheduler.schedule`](vllm/v1/core/sched/scheduler.py#L439)。

## 2. 构造阶段与运行阶段

阅读时先把问题分为两类：

```text
构造阶段：系统由哪些部件组成？
LLM -> EngineArgs -> VllmConfig -> Executor -> Worker/Model
    -> KV Cache -> Scheduler

运行阶段：一个请求如何流动？
Prompt -> Request -> Schedule -> Execute -> Update -> Output
```

推荐顺序：

1. 浅看初始化，画出对象关系。
2. 精读 `EngineCore.step()`。
3. 跟踪一次完整 prefill 和至少一次 decode。
4. 回头精读模型加载、显存探测、KV Cache 和 Scheduler 初始化。
5. 最后研究在线服务、多进程、分布式和高级功能。

建议投入约 30% 时间理解构造，70% 时间理解运行循环。

## 3. 关键对象关系

```text
LLM
└── LLMEngine
    ├── Renderer
    ├── InputProcessor
    ├── OutputProcessor
    └── EngineCoreClient
        └── EngineCore
            ├── ModelExecutor
            │   └── Worker
            │       └── ModelRunner
            │           └── torch.nn.Module
            └── Scheduler
                └── KVCacheManager
```

主要入口：

- [`LLM`](vllm/entrypoints/llm.py#L67)
- [`LLMEngine`](vllm/v1/engine/llm_engine.py#L48)
- [`EngineCoreClient`](vllm/v1/engine/core_client.py#L78)
- [`EngineCore`](vllm/v1/engine/core.py#L103)
- [`Scheduler`](vllm/v1/core/sched/scheduler.py#L69)
- [`Request`](vllm/v1/request.py#L59)
- [`SchedulerOutput`](vllm/v1/core/sched/output.py#L193)
- [`ModelRunnerOutput`](vllm/v1/outputs.py#L261)

仓库中同时存在 `vllm/engine` 和 `vllm/v1`。当前 `LLM` 已直接导入
V1 的 `LLMEngine`，学习时优先看 `vllm/v1`。

## 4. 最小调试场景

第一轮固定为一条“快乐路径”：

- 离线推理。
- Decoder-only 文本模型。
- 单请求、单 GPU。
- Greedy decoding。
- EngineCore 单进程。
- Eager execution。
- 暂时关闭 prefix caching。
- 不使用 LoRA、多模态、量化、推测解码、结构化输出和分布式。

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",
    tensor_parallel_size=1,
    enforce_eager=True,
    enable_prefix_caching=False,
)

outputs = llm.generate(
    ["Hello"],
    SamplingParams(temperature=0, max_tokens=2),
    use_tqdm=False,
)

print(outputs[0].outputs[0].text)
```

按照仓库规则使用 `uv` 和 `.venv/bin/python`：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
pre-commit install
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

调试时关闭 V1 EngineCore 多进程：

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  .venv/bin/python your_example.py
```

关闭后，`EngineCoreClient` 会选择
[`InprocClient`](vllm/v1/engine/core_client.py#L306)，其 `get_output()`
会在当前进程直接调用 `EngineCore.step()`，IDE 断点更容易命中。

`enforce_eager=True` 能减少 CUDA Graph 等执行分支，但初始化阶段仍会调用
`compile_or_warm_up_model()` 做必要的 warmup。

## 5. 真实离线调用链

```text
LLM.generate
  -> OfflineInferenceMixin._run_completion
    -> LLMEngine.add_request
      -> InprocClient.add_request
        -> EngineCore.add_request
          -> Scheduler.add_request
    -> OfflineInferenceMixin._run_engine
      -> LLMEngine.step
        -> InprocClient.get_output
          -> EngineCore.step
            -> Scheduler.schedule
            -> ModelExecutor.execute_model
            -> ModelRunner.execute_model
            -> ModelRunner.sample_tokens
            -> Scheduler.update_from_output
        -> OutputProcessor.process_outputs
```

源码入口：

- [`LLM.generate`](vllm/entrypoints/llm.py#L414)
- [`_run_completion`](vllm/entrypoints/offline_utils.py#L326)
- [`_run_engine`](vllm/entrypoints/offline_utils.py#L573)
- [`LLMEngine.add_request`](vllm/v1/engine/llm_engine.py#L218)
- [`LLMEngine.step`](vllm/v1/engine/llm_engine.py#L296)
- [`InprocClient.get_output`](vllm/v1/engine/core_client.py#L319)
- [`EngineCore.step`](vllm/v1/engine/core.py#L584)
- [`Scheduler.schedule`](vllm/v1/core/sched/scheduler.py#L439)
- [`Scheduler.update_from_output`](vllm/v1/core/sched/scheduler.py#L1670)

## 6. 第一层断点：完整主干

第一遍到达断点后，只看参数、关键状态和返回值，然后继续到下一个断点，
不要进入辅助函数。

| 顺序 | 断点 | 主要观察内容 |
| ---: | --- | --- |
| 1 | `LLM.generate` | 原始 prompt 与 `SamplingParams` |
| 2 | `OfflineInferenceMixin._run_completion` | 添加请求与运行引擎是两个阶段 |
| 3 | `LLMEngine.add_request` | 输入如何变成 `EngineCoreRequest` |
| 4 | `InprocClient.add_request` | Frontend/Core 边界 |
| 5 | `OfflineInferenceMixin._run_engine` | 离线 while 循环 |
| 6 | `LLMEngine.step` | 取 Core 输出并处理用户输出 |
| 7 | `InprocClient.get_output` | 单进程下如何触发 Core step |
| 8 | `EngineCore.step` | `schedule -> execute -> update` |
| 9 | `Scheduler.schedule` | 本轮选择的请求和 token 数 |
| 10 | `GPUModelRunner.execute_model` | SchedulerOutput 如何变成 GPU 输入 |
| 11 | `GPUModelRunner.sample_tokens` | Logits 如何变成 token |
| 12 | `Scheduler.update_from_output` | Token 如何写回 Request |

如果旧版 Model Runner 断点没有命中，当前配置可能选择了 Model Runner V2，
再检查
[`vllm/v1/worker/gpu/model_runner.py`](vllm/v1/worker/gpu/model_runner.py)。

## 7. `EngineCore.step()`：运行主干中心

优先精读
[`EngineCore.step`](vllm/v1/engine/core.py#L584)，它的核心只有四步：

```python
scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(scheduler_output, non_block=True)
model_output = future.result()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
```

建议观察：

```python
self.scheduler.has_requests()
scheduler_output.num_scheduled_tokens
scheduler_output.total_num_scheduled_tokens
scheduler_output.scheduled_new_reqs
scheduler_output.scheduled_cached_reqs
```

## 8. 第二层断点：Scheduler 状态推进

### 8.1 首次进入的 waiting 请求

- 从 waiting 队列取请求：
  [`scheduler.py`](vllm/v1/core/sched/scheduler.py#L697)
- 计算首次 `num_new_tokens`：
  [`scheduler.py`](vllm/v1/core/sched/scheduler.py#L879)
- 分配 KV Cache slots：
  [`scheduler.py`](vllm/v1/core/sched/scheduler.py#L973)
- 构造 `SchedulerOutput`：
  [`scheduler.py`](vllm/v1/core/sched/scheduler.py#L1208)
- 调度后推进状态：
  [`Scheduler._update_after_schedule`](vllm/v1/core/sched/scheduler.py#L1317)

### 8.2 后续 running 请求

- 遍历 running 请求：
  [`scheduler.py`](vllm/v1/core/sched/scheduler.py#L483)
- 计算 token 差值：
  [`scheduler.py`](vllm/v1/core/sched/scheduler.py#L516)
- 分配新增 KV slots：
  [`scheduler.py`](vllm/v1/core/sched/scheduler.py#L575)

### 8.3 Request 观察项

```python
request.request_id
request.status
request.num_prompt_tokens
request.num_tokens
request.num_computed_tokens
request.num_in_flight_tokens
request.num_output_placeholders
request.output_token_ids
request.max_tokens
```

`num_computed_tokens` 会在 Scheduler 产生本轮计划后乐观增加：

```python
request.num_computed_tokens += num_scheduled_token
request.num_in_flight_tokens += num_scheduled_token
```

因此它并不总表示 GPU 已完成计算。在异步调度和流水线并行场景下，
还要结合 `num_in_flight_tokens` 理解。

## 9. 第三层断点：模型执行与采样

旧版 `GPUModelRunner` 的建议观察点：

1. 准备模型输入：
   [`gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py#L4250)
2. 生成 `input_ids`、`positions` 等输入：
   [`gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py#L4401)
3. 模型 forward：
   [`gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py#L4450)
4. 计算 logits：
   [`gpu_model_runner.py`](vllm/v1/worker/gpu_model_runner.py#L4484)
5. 采样 token：
   [`GPUModelRunner.sample_tokens`](vllm/v1/worker/gpu_model_runner.py#L4553)
6. Scheduler 读取 sampled tokens：
   [`scheduler.py`](vllm/v1/core/sched/scheduler.py#L1675)
7. 写回 Request 并检查停止条件：
   [`Scheduler._update_request_with_output`](vllm/v1/core/sched/scheduler.py#L2094)

建议观察：

```python
input_ids.shape
positions.shape
hidden_states.shape
sample_hidden_states.shape
logits.shape
generated_token_ids
request.output_token_ids
request.status
```

第一轮不要从 `_model_forward()` 继续进入每个 Transformer layer。

当前 Model Runner 将 forward 与 sampling 分开。`execute_model()` 可能保存
临时状态后返回 `None`，随后 `EngineCore` 再调用 `sample_tokens()`；
这是正常设计，不表示模型没有输出。

## 10. Prefill 与 decode 的预期状态变化

假设 prompt 有 `N` 个 token，`max_tokens=2`，且 prompt 不需要 chunking。

```text
创建 Request:
  status = WAITING
  num_computed_tokens = 0
  num_tokens = N
  output_token_ids = []

第 1 轮:
  num_scheduled_tokens = N
  调度后 num_computed_tokens = N
  完成 prompt prefill
  根据最后位置 logits 采样 token A
  output_token_ids = [A]
  num_tokens = N + 1

第 2 轮:
  num_scheduled_tokens = 1
  调度后 num_computed_tokens = N + 1
  使用 token A 做一次 decode
  采样 token B
  output_token_ids = [A, B]
  status = FINISHED_LENGTH_CAPPED
```

注意：

- 短 prompt 的完整 prefill 通常会在同一轮采样第一个输出 token。
- 如果 prompt 被分成多个 chunk，中间 prefill chunk 不产生 token；最后一个
  prefill chunk 才采样第一个 token。
- 普通 decode 每轮通常安排一个 token。

## 11. 输入与输出边界

### 11.1 输入类型逐级收窄

```text
str prompt
  -> EngineInput
  -> EngineCoreRequest
  -> Request
```

推荐入口：

- [`OfflineInferenceMixin._add_request`](vllm/entrypoints/offline_utils.py#L552)
- [`LLMEngine.add_request`](vllm/v1/engine/llm_engine.py#L218)
- [`InprocClient.add_request`](vllm/v1/engine/core_client.py#L327)
- [`Request.from_engine_core_request`](vllm/v1/request.py#L223)

### 11.2 离线输出只返回最终结果

离线 `_add_request()` 会设置：

```python
params.output_kind = RequestOutputKind.FINAL_ONLY
```

因此中间若干轮 `LLMEngine.step()` 返回空列表是正常的。EngineCore 可能已经
生成 token，只是离线 API 暂时不把中间结果交给用户。

推荐观察：

- [`OutputProcessor.process_outputs` 调用处](vllm/v1/engine/llm_engine.py#L309)
- [`_run_engine` 收集 finished output](vllm/entrypoints/offline_utils.py#L594)

## 12. 初始化阶段的阅读重点

运行主循环看懂后，再按下面顺序回看初始化：

```text
LLM.__init__
  -> EngineArgs
  -> EngineArgs.create_engine_config
  -> VllmConfig
  -> Executor.get_class
  -> LLMEngine
  -> EngineCore
  -> ModelExecutor / Worker / Model
  -> KV Cache memory profiling and allocation
  -> Scheduler
  -> compile / warmup
```

建议断点：

| 断点 | 回答的问题 |
| --- | --- |
| [`LLM.__init__`](vllm/entrypoints/llm.py#L177) | 用户构造入口是什么 |
| [构造 `EngineArgs`](vllm/entrypoints/llm.py#L295) | 参数如何统一收集 |
| [`LLMEngine.from_engine_args`](vllm/v1/engine/llm_engine.py#L161) | 如何得到最终配置和 Executor |
| [选择 Executor](vllm/v1/executor/abstract.py#L48) | 单卡、多卡和 Ray 后端如何选择 |
| [创建 ModelExecutor](vllm/v1/engine/core.py#L131) | Worker/模型执行体系从哪里产生 |
| [获取 KV Cache 规格](vllm/v1/engine/core.py#L257) | 模型各层需要什么缓存 |
| [探测可用显存](vllm/v1/engine/core.py#L293) | KV Cache 容量如何确定 |
| [初始化实际 KV Cache](vllm/v1/engine/core.py#L330) | Cache 在 Worker 中如何创建 |
| [Compile/warmup](vllm/v1/engine/core.py#L332) | 启动阶段为何耗时 |
| [创建 Scheduler](vllm/v1/engine/core.py#L160) | Scheduler 如何得到 KV 配置 |

构造阶段最值得理解的顺序：

1. 解析配置并选择 Executor。
2. 创建 Worker 并加载模型。
3. 获取模型需要的 KV Cache 规格。
4. Profile 模型峰值显存，计算可用于 KV Cache 的空间。
5. 创建实际 KV Cache。
6. 用 KV Cache 配置创建 Scheduler。
7. 编译或 warmup 模型。

第一轮只认识这些核心配置：

- `model_config`
- `cache_config`
- `scheduler_config`
- `parallel_config`
- `compilation_config`

`EngineArgs.create_engine_config()` 中的平台兼容、校验和高级配置分支按需
回看，不必第一次全部读完。

## 13. 第一轮暂时跳过的分支

遇到以下关键词时，先记录到“支线清单”，不要立即展开：

- `has_encoder_inputs`：多模态或 encoder。
- `spec_token_ids`：推测解码。
- `lora_config`：LoRA。
- `connector`、`remote_kv`：远程 KV 和分离式部署。
- `throttle_prefills`：数据并行负载协调。
- `next_decode_eligible_step`：异步调度和流水线并行。
- `mamba`：混合状态模型。
- `grammar`：结构化输出。
- `preempted_reqs`：显存不足时的抢占。
- CUDA Graph、`torch.compile` 和 kernel 细节。

第一轮 Scheduler 只理解：

```text
从 running / waiting 取请求
  -> 计算 num_new_tokens
  -> 检查 token budget
  -> allocate_slots 分配 KV 空间
  -> 构造 SchedulerOutput
  -> 执行后更新 Request
```

## 14. 用测试作为可执行文档

安装测试依赖：

```bash
uv pip install -r requirements/test/cuda.in
```

建议先阅读或运行：

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py \
  -k test_schedule_order -v

.venv/bin/python -m pytest \
  tests/v1/core/test_kv_cache_utils.py \
  -k test_free_kv_cache_block_queue_operations -v
```

推荐测试入口：

- [`test_scheduler.py`](tests/v1/core/test_scheduler.py)
- [`test_kv_cache_utils.py`](tests/v1/core/test_kv_cache_utils.py)
- [`test_single_type_kv_cache_manager.py`](tests/v1/core/test_single_type_kv_cache_manager.py)
- [`test_llm_engine.py`](tests/v1/engine/test_llm_engine.py)
- [`test_output_processor.py`](tests/v1/engine/test_output_processor.py)

阅读顺序：

1. 先读测试名和断言。
2. 明确输入、可观察输出和状态变化。
3. 再回到实现寻找完成该行为的最短路径。

## 15. 三遍阅读法

### 第一遍：骨架

- 只看方法签名、关键参数和返回值。
- 把子模块暂时当作黑盒。
- 画出请求生命周期和对象持有关系。

### 第二遍：状态

- 跟踪 `Request` 的字段变化。
- 分别观察一次完整 prefill 和一次 decode。
- 理解 `SchedulerOutput` 如何驱动 Model Runner。

### 第三遍：机制

按以下顺序逐个打开黑盒：

1. KV block 分配。
2. Continuous batching。
3. Prefix caching。
4. 抢占。
5. Model Runner 输入构造。
6. Attention backend。
7. CUDA Graph 和 `torch.compile`。
8. TP、PP、DP。
9. Speculative decoding。

每次只研究一个能力，比较它给主路径增加了哪些状态和分支。

## 16. 第一阶段完成标准

能够回答以下问题，就已经抓住主干：

- 字符串 prompt 在哪里变成内部 `Request`？
- `Request` 如何从 waiting 转为 running？
- Scheduler 如何计算本轮 `num_new_tokens`？
- KV Cache 容量为什么会限制可调度 token 数？
- `SchedulerOutput` 如何变成 GPU 输入？
- 完整 prefill 为什么可以直接产生第一个输出 token？
- 采样 token 在哪里写回 `Request`？
- `num_computed_tokens` 与 `num_in_flight_tokens` 有什么区别？
- 请求何时进入 finished 状态？
- 为什么离线模式的中间 step 可能没有用户可见结果？
- 为什么关闭 EngineCore 多进程后断点更容易命中？
- 初始化为什么要先加载和 profile 模型，再决定 KV Cache 容量？

最重要的检验是：进入下一轮 `EngineCore.step()` 前，能够预测本轮大概会
调度多少 token、请求状态如何变化，以及下一步会进入哪个组件。

