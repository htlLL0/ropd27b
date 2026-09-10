# 在单张 B200 上生成数据、训练 clean-T+ OPCD 并完成 AgentDojo 评测

按本 README 操作即可。**仓库已包含代码、输入数据和模型配套文件；在 B200 服务器上从上游开源模型仓库下载 18 个权重分片。全部日志、生成结果和训练 checkpoint 统一保存在仓库的 `output/` 中。**

新增无需标注的训练路线：**T+ 看到原始干净输入，T− 和 student 看到同一份带攻击输入；全部 9,600 条进入 OPCD，不做正常样例 SFT。** 先完成下面 4 步生成，再执行“无需标注，直接训练”中的命令。已有本包完整生成结果时可直接进入训练。

训练完成后可运行“AgentDojo：原始模型与训练模型对照测试”。测试代码和数据均已包含，使用 AgentDojo 自带判定；原始结果之外，自动生成每份不超过 **90,000 字节**、带完整性校验的 TXT 传输版本。

**完整流程按阶段串行进行：生成 9,600 条 → OPCD 训练 → 原始模型 AgentDojo → 训练模型 AgentDojo → 指标汇总与 TXT 导出。当前需要分别启动生成、训练、评测三段，并等待上一段完成及容器退出后再启动下一段；尚未提供一条命令自动衔接三段的总入口。** 对应入口依次为 `./docker_b200.sh start --gpu 0`、`./train_b200.sh start --gpu 0`、`./agentdojo_b200.sh start --gpu 0`。这三条后台启动命令不能连续粘贴执行；评测一旦启动，会自动依次完成两个模型的测试和结果导出。阶段内部的生成并发仍为 16，AgentDojo 为单 worker。

固定配置：Qwen3.8-27B、单张 B200、BF16、TP=1、并发 16、温度 1.0、`top_p=1.0`、thinking 开启，每条最多生成 16,384 tokens，上下文上限 32,768。无需修改配置或重新生成输入数据。

## 先确认服务器环境

推荐使用 Docker。目标服务器需要：

- Linux x86_64、Git、Python 3.10 或更新（带 `venv` 模块，用于安装下载工具）。
- 一张完整、空闲的 NVIDIA B200，NVIDIA 驱动支持 CUDA 13（建议 580 或更新，以预检为准）。
- Docker 和已配置的 NVIDIA Container Toolkit，当前账号能够运行 GPU 容器。
- 磁盘建议准备 **500 GB 可用空间，优先使用 SSD**；保留多轮运行或多个完整 TXT 快照时，建议 **1 TB 可用空间**。这是按当前文件及代码估算的容量规划，尚未测得 B200 完整运行的磁盘峰值。
- 首次能够访问 GitHub、Hugging Face、PyPI 和 Docker 镜像仓库；权重、依赖和镜像准备好后，生成过程不需要联网。

空间构成：原始模型及静态数据约 **55.85 GB**，评测时独立合并模型约 **56 GB**；当前三种 Docker 镜像共享基础层，最终镜像层合计约 **20.7 GB**，不能简单将三个镜像显示大小相加。上述固定部分合计约 **133 GB**，另外还需容纳镜像下载/构建临时文件、9,600 条生成数据及汇总、训练数据、LoRA/优化器 checkpoint、缓存和评测完整日志。训练保留最近两个完整 checkpoint。TXT 导出期间，原始评测日志之外还会短暂同时存在一份拼接临时文件和一份分片副本；每个保留的旧 snapshot 也会继续占空间。500 GB 是建议预算，不是对任意日志长度和快照数量的硬上限保证。

如果 Docker 数据目录与仓库位于不同分区，分别检查两边的可用空间；仓库所在数据盘空间充足，并不能保证 Docker 所在系统盘够用。查看命令：`df -h .`，以及 `docker info --format '{{.DockerRootDir}}'` 后对返回目录执行 `df -h`。

在 **B200 服务器的终端**检查：

```bash
python3 --version
git --version
nvidia-smi
docker info
```

Docker 方式无需在宿主机安装 PyTorch、Transformers 或 CUDA Toolkit。若不能使用 Docker，完成下面第 1、2 步后，转到本 README 的“没有 Docker 时”一节。

## 首次运行：4 步

### 1. 在 B200 服务器上克隆仓库

```bash
git clone https://github.com/htlLL0/ropd27b.git "$HOME/ropd27b"
cd "$HOME/ropd27b"
```

后续命令均在这个目录中执行。`inputs/`、`source_data/`、tokenizer 和配置已经齐全，无需 Git LFS。

### 2. 在 B200 服务器上从开源仓库下载权重

权重来源是 [Qwen/Qwen3.8-27B 官方开源仓库](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)，固定 revision 为 **`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`**。它与此前 SecOPD 项目目录中使用的模型来源和权重一致，18 个分片共约 55.6 GB。

在 B200 服务器的 `$HOME/ropd27b` 目录中执行以下整段命令。它使用独立虚拟环境安装官方 `hf` 下载工具，直接下载到 `model/`，随后完整校验文件：

```bash
(
  set -euo pipefail
  mkdir -p output/logs output/setup
  python3 -m venv .venv

  PIP_CACHE_DIR="$PWD/output/setup/pip-cache" \
    .venv/bin/python -m pip install "huggingface_hub==1.8.0" \
    2>&1 | tee -a output/setup/install_hf.log

  HF_HOME="$PWD/output/setup/huggingface" HF_HUB_DOWNLOAD_TIMEOUT=60 \
    .venv/bin/hf download Qwen/Qwen3.8-27B \
      --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
      --include 'model-*-of-*.safetensors' \
      --local-dir ./model \
    2>&1 | tee -a output/logs/download_weights.log

  ./run_b200.sh verify
)
```

下载日志为 `output/logs/download_weights.log`，工具安装日志为 `output/setup/install_hf.log`，文件校验日志为 `output/logs/native_verify.log`。断网或下载失败后，网络恢复时重新执行本步骤；保留已下载文件和下载元数据。

命令只下载权重，不覆盖仓库中的 tokenizer、配置或权重索引。固定 revision 下的 18 个分片已与本包的大小和 SHA256 逐一核对；看到校验输出 `"status": "pass"` 后继续第三步。官方命令说明见 [Hugging Face 下载文档](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download)。

### 3. 下载运行镜像并预检 B200

```bash
./docker_b200.sh pull
./docker_b200.sh preflight --gpu 0
```

`--gpu 0` 表示编号为 0 的 B200；如果实际编号不同，后续启动、恢复命令也使用该编号。只选择一张空闲的完整 B200。

**预检成功，输出包含 `"bf16_probe": "pass"` 后再继续。** 预检会检查固定依赖、单卡可见性和实际 BF16 运算。

如需提前单独检查所有文件及 9,600 条输入，可额外执行 `./docker_b200.sh check`。该命令不申请 GPU，但会完整读取权重计算 SHA256，可能需要几分钟；正式启动也会执行这些检查。

### 4. 启动生成并查看状态

```bash
./docker_b200.sh start --gpu 0
./docker_b200.sh status
```

任务在后台运行，退出 SSH 不会停止 Docker 任务。启动后自动校验文件和全部输入，加载模型，先生成固定的 **12 条真实验证样本**；通过后自动继续其余 **9,588 条**，完成审计和汇总后释放所属 GPU。12 条计入总数 9,600，无需再次启动。

`start` 返回 `launched_preflight_pending` 只表示后台任务已创建；实际进度以 `status` 为准。首次权重校验可能持续几分钟。

## 查看进度和日志

```bash
# 每 5 秒刷新进度；时间按北京时间显示
watch -t -n 5 ./docker_b200.sh status

# 主程序日志
./docker_b200.sh logs

# 模型加载和推理日志
tail -n 100 -F output/main/server.log
```

没有 `watch` 时，直接执行 `./docker_b200.sh status`。在查看命令中按 `Ctrl+C` 只退出查看，不会停止生成。ETA 在本次运行完成至少 100 条后开始显示。

| 状态 | 含义 |
|---|---|
| `preflight` | 文件、环境和输入检查 |
| `model_loading` | 正在加载模型 |
| `smoke` / `bulk` | 最先 12 条真实验证 / 全量生成 |
| `draining` / `paused` | 正在保存已有请求 / 已干净暂停 |
| `generation_complete` | 全部生成及最终审计完成 |
| `failed` | 有异常，查看日志和错误回执 |

## 结果在哪、如何确认完成

默认所有结果位于 `$HOME/ropd27b/output/`：

```text
output/
├── logs/                    # 权重/镜像下载、校验、预检、启动、暂停等日志
├── setup/                   # 下载工具/原生依赖安装日志及缓存
└── main/                    # 默认任务目录
    ├── supervisor.log       # 主程序 stdout/stderr 和异常堆栈
    ├── server.log           # 模型服务日志
    ├── progress.json        # 实时进度
    ├── completed.jsonl      # 已完成索引日志，不是完整轨迹汇总
    ├── cases/               # 逐条请求、原始响应、轨迹、执行及错误记录
    ├── generations.jsonl    # 9600 条全部完成后的完整轨迹汇总
    ├── generation_summary.json
    ├── completion.json      # 完成回执
    ├── cache/               # 运行缓存
    └── tmp/                 # 临时文件
```

`cases/` 会持续保存逐条结果；`generations.jsonl` 仅在全部完成并通过审计后出现。容器退出后，日志和结果仍保存在宿主机。

以下条件同时满足才算完成：`status` 显示 `generation_complete` 和 `9600/9600`；完成回执中 `status=complete`、`samples=9600`、`server_released=true`；汇总文件有 9,600 行。

```bash
cat output/main/completion.json
wc -l output/main/generations.jsonl
```

完成后复制整个 `output/` 即可带走日志、结果和回执。生成文件中的 `training_ready=false` 是原收集流程的通用标记；新增的 clean-T+ 路线会独立审计原始响应并创建自己的训练准入回执，**无需补 A/U 标注**。生成任务完成后退出，训练需执行下一节的命令。

## 无需标注，直接训练

### 1. 准备训练镜像并检查

在同一仓库目录执行；首次构建需要访问 PyPI，随后训练不联网。镜像在上述固定 vLLM/CUDA 13 镜像上增加 `peft==0.20.0`，不升级原有 PyTorch/Transformers。构建不会把 56 GB 模型发送给 Docker。

```bash
./train_b200.sh build
./train_b200.sh check
./train_b200.sh preflight --gpu 0
```

`check` 不占 GPU：检查 9,600 对真实 clean/attacked 输入，以及小型随机初始化 Qwen 混合注意力模型的反向传播、冻结 teacher、梯度一致性与保存恢复。日志保存在 `output/logs/training_check.log`。它不等于 27B 在 B200 上的实测。

### 2. 生成完成后启动训练

确认 `./docker_b200.sh status` 为 `generation_complete`，生成容器已经退出，再执行：

```bash
./train_b200.sh start --gpu 0
./train_b200.sh status
watch -t -n 5 ./train_b200.sh status
```

默认读取 `output/main/`，后台训练写入 `output/clean_opcd/`。退出 SSH 后 Docker 训练继续运行。启动后先重新审计全部 raw response、token、来源与汇总文件，然后加载模型。**第一步实际完成 27B 的前向、反向、LoRA 更新及 checkpoint 保存，通过后自动继续余下训练。** 第一步计入总数 9,600，不额外生成、不做试验数据混入。

```bash
# 训练日志
./train_b200.sh logs

# 真实第一步检查结果（第一次更新完成后出现）
cat output/clean_opcd/first_update_gate.json

# 最终回执和最近的 checkpoint（训练完成后检查）
cat output/clean_opcd/completion.json
cat output/clean_opcd/latest_checkpoint.json
```

完成标准：状态 `training_complete`、`9600/9600`，`completion.json` 中 `all_samples_opcd=true`、`semantic_labels_used=false`、`clean_sft_used=false`，随后容器退出并释放显卡。训练 ETA 在本次会话完成 10 次更新后按实测速率显示；长短样本差异较大，ETA 仅作参考。

### 3. 暂停和恢复

```bash
./train_b200.sh pause
./train_b200.sh status
# 等到 paused 且容器 Running=false 后恢复
./train_b200.sh start --gpu 0 --resume
```

暂停会等待当前更新完成，并保存 LoRA、优化器、随机状态和训练位置。第一步、每 100 步、正常暂停及最终步骤保存 checkpoint，保留最近两个完整版本。异常退出时保留已有数据与 checkpoint；恢复从最近完整 checkpoint 继续，尚未保存的更新会重新计算。不要在同一个运行目录里修改配置、代码或输入。出现错误查看 `output/clean_opcd/failure.json` 和 `supervisor.log`。

若生成任务名称不是 `main`，或需要另开训练，所有管理命令使用对应名称：

```bash
./train_b200.sh start --gpu 0 --generation-run-id run2 --run-id clean_opcd_run2
./train_b200.sh status --run-id clean_opcd_run2
```

### 训练配置和数据流

| 项目 | 本版本行为 |
|---|---|
| T+ 输入 | 原始 `user_query + clean_context`；不带注入，不使用 gold answer 或原 quarantine 视图 |
| T− / student 输入 | 相同的原版 `user_query + contaminated_context` |
| 三方回答序列 | 同一条实际采样的 student response，含 thinking、final 和实际返回的结束 token；teacher 不另生成答案 |
| 样例 | 9,600 条全部参与，一次固定随机打乱、一轮训练；不按 A/U、风险标签、是否攻击成功筛选 |
| 损失 | 保留现有全词表 T+/T− OPCD 方向、deficit 和 Hybrid-KL；`eta=0.7`、`delta_max=2`、forward/reverse 各 0.5 |
| 标注 / 正常 SFT | 不需要标注，`attack_gate=1`、`c=1`、SFT 权重为 0；不读取 parent utility controls，不做 c 标定 |
| token 门控 | 保留 OPCD 原有 disagreement/deficit 门控；无须修正的 token 可以贡献零损失，不转为 SFT |
| LoRA / 优化器 | rank 8、alpha 16、dropout 0；Qwen 文本全注意力层的 q/k/v/o；AdamW，LR `1e-5`，梯度裁剪 1 |
| 单卡内存 | 共用一个冻结 BF16 底座；两 teacher 关闭 adapter 并顺序前向；student 梯度 checkpoint；词表投影按 16 tokens 分块、一次 decoder 反向 |
| 长度 | 上限 32,768，禁止静默截断；达到生成上限的真实响应也做 OPCD，不补造 EOS，不按答案格式删样本 |

完整配置见 [config/clean_teacher_opcd.json](config/clean_teacher_opcd.json)，实现边界见 [provenance/clean_teacher_opcd_contract.json](provenance/clean_teacher_opcd_contract.json)。LoRA 使用 [PEFT 官方实现](https://huggingface.co/docs/peft/package_reference/lora)。当前训练镜像使用 Transformers 的 PyTorch 线性注意力实现，未额外编译 FLA/causal-conv1d；实际 27B 长序列速度与显存以目标机第一步和后续日志为准，OOM 会停止并保留 checkpoint，不自动缩短序列。

**本版沿用“先生成 9,600 条，再训练”的固定轨迹流程。** clean teacher 与完整 response 对齐方式参考 SecOPD，但保留 T− 和 OPCD 损失；它不是每步更新 student 后重新采样的严格在线 SecOPD。现有仅生成入口继续可用，原数据、采样温度和旧 teacher views 保留。

```text
output/
├── logs/training_*.log        # 镜像构建、检查、预检、启动和暂停日志
├── setup/training_tests/     # CPU 验证临时目录
├── main/                    # 原始 9600 条生成结果
└── clean_opcd/
    ├── supervisor.log       # 完整训练日志和异常
    ├── training_samples.jsonl
    ├── input_gate.json      # 本路线的输入与原始响应审计
    ├── first_update_gate.json
    ├── progress.json
    ├── metrics.jsonl        # 更新日志；异常恢复可能重算未提交的更新
    ├── latest_checkpoint.json
    ├── checkpoints/         # adapter_model.safetensors、配置、optimizer/RNG、游标与校验
    ├── completion.json
    ├── cache/
    └── tmp/
```

最终产物是 **LoRA adapter**。使用时仍需本 README 固定 revision 的基础权重；adapter 路径由 `latest_checkpoint.json` 的 `path` 给出，相对于 `output/clean_opcd/`。

## AgentDojo：原始模型与训练模型对照测试

### 1. 准备评测环境

下面命令在 B200 服务器的仓库目录执行。评测镜像单独构建，使用固定 CUDA 13 / vLLM 基础镜像，并安装 AgentDojo 和 PEFT；首次构建需要联网。后续评测在无外网的容器中执行，不需要 OpenAI、Claude 等服务的 Key，也不调用付费 judge。

```bash
./agentdojo_b200.sh build
./agentdojo_b200.sh check
./agentdojo_b200.sh preflight --gpu 0
```

AgentDojo 固定版本为 **v1.2.1**，源码 revision 为 `d3640b5b03a88eb44dad96852e1de1ef437d836e`。`third_party/agentdojo/` 包含官方完整源码、四套环境、任务、注入向量、判定函数和许可；`inputs/agentdojo_cases.jsonl` 固定全部测试 ID、任务和静态攻击内容，无需另行下载测试数据。

`check` 不使用 GPU，检查官方源码哈希、完整测试清单、原生判定、TXT 分片还原，并运行小模型 adapter 合并检查。它只验证工程链路，不是原始模型或训练模型的正式评测结果。

### 2. 一次启动，依次测试两个模型

确认生成和训练任务均已退出、B200 空闲，`output/clean_opcd/completion.json` 显示完成后执行：

```bash
./agentdojo_b200.sh start --gpu 0
./agentdojo_b200.sh status
watch -t -n 5 ./agentdojo_b200.sh status
```

默认先测试 **原始 Qwen3.8-27B**，释放模型服务，再把完成训练的最终 LoRA 合并到独立模型目录并测试 **训练后的模型**。基础模型文件不变。合并产物约 56 GB，合并时磁盘还需至少额外 10 GiB 余量；日志和 TXT 副本也会继续占空间。合并只在服务器实际执行本命令后进行。

若想在训练前先测原始模型，可单独执行：

```bash
./agentdojo_b200.sh start --gpu 0 --models base --run-id agentdojo_base
./agentdojo_b200.sh status --run-id agentdojo_base
```

单独的 base 任务不需要训练 checkpoint；默认双模型任务会完整测试两者，不会把单独任务的旧结果静默混入。如果训练目录不是 `clean_opcd`，在启动、恢复时加 `--training-run-id 对应目录名`。

### 3. 判定口径与测试规模

| Suite | 正常任务 | 攻击组合 | 攻击目标可执行性检查 |
|---|---:|---:|---:|
| workspace | 40 | 560 | 14 |
| banking | 16 | 144 | 9 |
| travel | 20 | 140 | 7 |
| slack | 21 | 105 | 5 |
| 每个模型合计 | **97** | **949** | **35** |

两个模型共 **2,162 条完整任务轨迹**。35 条检查是 AgentDojo 原有的 injection-task-as-user-task 诊断，单列汇报，不混入正常成功率或 ASR 的分母。

**成功与否由未修改的 AgentDojo 判定函数负责。** 入口调用其 `run_task_without_injection_tasks()` / `run_task_with_injection_tasks()`，底层使用任务类 `utility()` / `utility_from_traces()` 和 `security()` / `security_from_traces()`，根据回答、工具调用和环境状态判定。`security=true` 在这里表示**攻击成功**。源码来源见 [固定 AgentDojo 仓库](https://github.com/ethz-spylab/agentdojo/tree/d3640b5b03a88eb44dad96852e1de1ef437d836e) 与 [provenance/agentdojo_source.json](provenance/agentdojo_source.json)。

汇总分别给出：正常任务成功率、受攻击任务成功率、ASR、每套 suite 的结果，以及训练模型相对原始模型的百分点变化。完整结果要求所有对应任务完成且没有基础设施错误；部分运行只显示已完成数量和诊断比例。

AgentDojo 上游在某些上下文溢出/服务异常中会回填 `utility=false, security=true`。本包完整保留原始值和 `official_raw_asr`，同时记录 `infrastructure_errors`；发生异常时，正式 `asr` / `task_success_rate` 留空，已完成有效样本的诊断比例单列，避免把服务故障解释成模型安全结论。

两模型使用同一套 `important_instructions` 攻击，攻击阶段使用 `repeat_user_prompt`，工具结果统一用 `input` role；thinking 开启、总上下文 32,768、单 worker。**评测温度为 0.0、top_p=0.9，不额外设单次输出上限**，与此前 SecOPD 评测方式对齐。训练数据生成的温度仍为 1.0。每个任务/请求的 seed 固定且在两个模型间配对。完整参数见 [config/agentdojo.json](config/agentdojo.json)。

### 4. 结果、日志与暂停恢复

```bash
./agentdojo_b200.sh logs
cat output/agentdojo/RESULTS.txt
cat output/agentdojo/summary.json
cat output/agentdojo/completion.json
```

正常完成时状态为 `evaluation_complete`，进度 `2162/2162`，汇总 `status=complete`，两个模型各有 1,081 个任务结果。随后容器退出，所属 GPU 释放。

```bash
./agentdojo_b200.sh pause
./agentdojo_b200.sh status
# 等到 paused 且容器 Running=false 后恢复
./agentdojo_b200.sh start --gpu 0 --resume
```

暂停会等当前 AgentDojo 任务完成，然后释放服务并生成当前进度的 TXT 副本；长任务可能需要等待多次模型调用。基础设施错误会保存原始错误并停止。显式 `--resume` 会校验已成功任务的文件哈希，保留旧失败/未完成 attempt，在新 attempt 中重跑该任务；不会重跑已完成的有效任务。模型、输入和配置必须保持一致。

```text
output/agentdojo/
├── RESULTS.txt                    # 便于直接查看的指标摘要
├── summary.json                   # 原始模型/训练模型、各套任务、配对比较
├── run_binding.json               # 固定模型、checkpoint、任务与配置身份
├── protocol.json / environment.json / hardware.json
├── supervisor.log / merge.log     # 主流程与 adapter 合并日志
├── checkpoint_identity.json       # 最终 checkpoint 与 adapter 哈希
├── merge_receipt.json             # 合并模型各文件的哈希
├── base/                          # 原始模型
│   ├── server.log / server_command.json
│   └── cases/<case_id>/attempt_*/
│       ├── result.json            # 官方判定值及文件完整性记录
│       ├── traces/                # AgentDojo 原生完整轨迹
│       └── requests/              # 每次请求、原始响应、错误和耗时
├── trained/                       # 训练模型，结构同 base
├── adapter_snapshot/              # 固定 adapter 副本
├── merged_model/                  # 独立合并权重，不修改原始模型
├── transfer_latest.json           # 最近一次 TXT 传输副本的位置
└── transfers/snapshot_*/           # 可直接传走的 TXT 文件
```

### 5. TXT 传输版本：每份不超过 90 KB

**评测完成、正常暂停或捕获到运行失败后，都会额外生成 TXT 版本，原始文件继续保留。** 副本包含结果、原生轨迹、每次完整请求/响应、服务日志、主程序日志、错误、配置、环境和 checkpoint 身份。权重、adapter 二进制、模型配套文件的重复副本、缓存和临时文件不放入传输包。

查看最新副本目录：

```bash
cat output/agentdojo/transfer_latest.json
./agentdojo_b200.sh status
```

每个 `snapshot_*` 文件夹内包含一个 `000000_CONTROL.txt` 和多个 `part-000001-of-XXXXXX.txt`。**包括文件头在内，每个文件最多 90,000 字节**，比 90 KiB 更严格。文本内容保留 UTF-8，可读；分片不会切断汉字的 UTF-8 编码。摘要排在最前面，完整过程日志随后。文件很多时，按同一 snapshot 的顺序逐批传输即可。

传走**同一个 snapshot 目录里的全部 TXT 文件**，不要混合不同 snapshot。保持文件名与内容，接收端只需要 Python 3 和仓库中的 `scripts/txt_transfer.py`：

```bash
# 假设收到的全部 TXT 放在 ./received_txt
# 先检查缺片、重复、顺序、内容变化及所有原始文件的 SHA256
python3 scripts/txt_transfer.py verify --source ./received_txt

# 拼接为一个可读的大 TXT；会先校验，再生成
python3 scripts/txt_transfer.py join \
  --source ./received_txt --destination ./agentdojo_joined.txt

# 或恢复完整原目录结构，JSON、日志、轨迹逐字节还原
python3 scripts/txt_transfer.py restore \
  --source ./received_txt --destination ./agentdojo_restored
```

控制算法采用 **分片编号和总数 + 每片 SHA256 + 有序哈希链 + 整体数据流 SHA256 + 每个原文件的大小与 SHA256**。缺片、重复、错序、混入另一包、截断或内容损坏都会报错；校验失败不发布拼接/还原结果，也不覆盖已有目标目录。这个校验保证传输完整性，部分或失败的评测不会因此变成完整结果。

如果需要重新生成副本，在评测容器已经停止后执行：

```bash
./agentdojo_b200.sh export
```

每次生成独立的新 snapshot，旧副本保留。TXT 副本包含完整日志，会额外占用磁盘；后续分析直接使用还原目录即可。

## 暂停、恢复和另开任务

```bash
# 请求暂停并检查状态
./docker_b200.sh pause
./docker_b200.sh status
```

脚本停止新增请求，等待已有请求保存，再退出模型服务。长请求可能需要等待，单请求超时为 3,600 秒。确认 `paused`、在途数为 0，且容器 `Running=false` 后，再备份或恢复。

```bash
# 恢复同一任务
./docker_b200.sh start --gpu 0 --resume
```

已完成记录会先校验再跳过，不会重复生成。存在未完成或带错误的 case 时，会拒绝静默重跑；先检查对应请求和错误记录。运行中不要更新代码、配置、输入或校验清单。

如需另开独立任务，在空闲 B200 上使用新的名称：

```bash
./docker_b200.sh start --gpu 0 --run-id run2
./docker_b200.sh status --run-id run2
```

结果将保存到 `output/run2/`；日志查看、暂停和恢复也要加上同一个 `--run-id run2`。

## 没有 Docker 时

先完成前面的仓库克隆和开源权重下载。使用带 `venv` 模块的 Python 3.12，安装依赖需要联网；驱动和 B200 的要求相同。

```bash
./install_native.sh
./run_b200.sh preflight --gpu 0
# 预检通过后启动
./run_b200.sh start --gpu 0
watch -t -n 5 ./run_b200.sh status
```

依赖安装在仓库的 `.venv/`，安装日志为 `output/setup/install_native.log`，其余结果布局与 Docker 相同。

如果还要原生运行 clean-T+ 训练，在已有依赖基础上安装 PEFT，然后前台运行；请放在 `tmux` 或平台持久任务中：

```bash
(
  set -euo pipefail
  PIP_CACHE_DIR="$PWD/output/setup/pip-cache" \
    .venv/bin/python -m pip install --no-deps peft==0.20.0 \
    2>&1 | tee -a output/setup/install_training.log
)
.venv/bin/python -u -B scripts/train_clean_teacher.py run --gpu 0
# 通过 Ctrl+C 请求保存后暂停；确认进程退出后恢复：
.venv/bin/python -u -B scripts/train_clean_teacher.py run --gpu 0 --resume
```

训练输出仍为 `output/clean_opcd/`，可直接读取其中的 `progress.json`；不要混用原生和 Docker 管理同一个训练目录。

```bash
# 查看主程序日志
tail -n 100 -F output/main/supervisor.log

# 暂停
./run_b200.sh pause
./run_b200.sh status

# 确认 paused、在途数为 0 且进程退出后恢复
./run_b200.sh start --gpu 0 --resume
```

不要混用 Docker 和原生方式管理同一个任务。若服务器会在 SSH 退出时清理用户进程，在 `tmux` 或平台持久任务中前台执行 `./run_b200.sh run --gpu 0`。已有完全匹配依赖的环境可通过 `OPCD_PYTHON=/path/to/python` 指定。

## 出错时先看这里

| 现象 | 操作 |
|---|---|
| Docker 不可用或权限不足 | 检查 Docker 服务和账号权限；无法使用时改用原生方式 |
| 创建虚拟环境提示缺少 `venv` 或 `ensurepip` | 安装与当前 Python 匹配的 venv 组件后重试第二步；Ubuntu/Debian 通常为 `python3-venv` |
| 权重下载连接失败或超时 | 检查目标服务器到 Hugging Face 的连接及代理，查看 `output/logs/download_weights.log`，恢复网络后重试第二步 |
| 找不到 `nvidia` runtime 或 GPU | 检查 NVIDIA Container Toolkit 是否已为 Docker 配置 |
| `B200 required` / GPU 已被占用 | 核对 `nvidia-smi`，选择正确且空闲的完整 B200 |
| 文件缺失、软链接、大小或 SHA256 不匹配 | 检查 18 个权重是否下载完整、revision 是否正确，以及是否混用了不同版本文件 |
| `Environment differs from pinned image` | 使用脚本指定的固定镜像，或用原生安装脚本建立匹配环境 |
| 已有任务或 `FileExistsError` | 查看现有状态；干净暂停后用 `--resume`，独立任务用新的 `--run-id` |
| 主程序日志尚未创建 | 查看 `output/logs/docker_start.log` 或 `output/logs/native_start.log` |
| 加载很久或状态为 `failed` | 查看 `output/main/supervisor.log`、`server.log`、`failure.json` 和对应 case 的 `error.json` |
| 未完成 case 导致无法恢复 | 保留整个 `output/`，核对已保存的请求、响应和错误，不删除记录来绕过检查 |
| 外部 Docker daemon 找不到挂载路径 | 将 `OPCD_DOCKER_BUNDLE` 设为 Docker 宿主机可见的仓库绝对路径；普通宿主机运行不需要设置 |

## 固定版本和验证范围

运行参数见 [config/generation.json](config/generation.json)，文件大小与 SHA256 见 [bundle_manifest.json](bundle_manifest.json)。固定 Docker 镜像为：

```text
vllm/vllm-openai:v0.24.0@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f
```

镜像包含 PyTorch 2.11.0+cu130、CUDA 13.0、vLLM 0.24.0、Transformers 5.12.1，并含 B200 的 `sm_100` 编译支持。

已完成全新克隆、全部权重哈希、9,600 条真实输入和 14 项 CPU 测试；使用明确标记的模拟响应完成了暂停、恢复及 9,600 条汇总流程。**目标 B200 的真实执行仍需该服务器的预检与 12 条真实生成验证通过。** 仿真数据不能用于训练或模型评估。

新增训练路线已通过 9,600 对原生 tokenizer 输入检查和 **8 项 CPU 测试**，包括 BF16 小型混合注意力模型更新、全词表分块梯度一致、teacher 冻结、checkpoint 完整性及恢复、拒绝仿真数据。普通用户权限的 Docker 入口也已检查。记录见 [validation/clean_teacher_training_verification.json](validation/clean_teacher_training_verification.json)。尚未在真实 B200 上运行 27B 训练，实际吞吐、长序列峰值显存与模型效果尚未验证。

AgentDojo 入口已通过 **16 项 CPU 测试**：完整测试清单与源码哈希、四套环境的原生判定、模拟 HTTP 请求经过真实评测器、小模型 adapter 合并、失败后自动导出、普通用户下的 vLLM 服务模块导入，以及 TXT 缺片/重复/损坏检测和逐字节还原。检查在无 GPU、普通用户权限的 Docker 中完成；A100 会在显存分配前被拒绝。记录见 [validation/agentdojo_delivery_verification.json](validation/agentdojo_delivery_verification.json)。**尚未运行 27B 模型的正式 AgentDojo 评测，没有实际 ASR 或成功率结果。** 镜像依赖检查拒绝新冲突，仅保留固定 vLLM 基础镜像原有的两条元数据警告，详情在验证记录中。

本项目的操作说明统一在本 README。模型卡保存在 [model/MODEL_CARD.md](model/MODEL_CARD.md)，许可见 [model/LICENSE](model/LICENSE)。`provenance/` 和 `validation/` 保留历史来源及验证记录，其中旧文件名或摘要对应当时的版本。
