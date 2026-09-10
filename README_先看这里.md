# 单张 B200：Qwen3.8-27B 生成 9,600 条训练候选轨迹

**按步骤操作请看 [使用说明.md](使用说明.md)**，其中包含首次启动、进度和日志查看、暂停恢复、完成判定及常见问题。下文保留配置和技术交接细节。

整个文件夹可复制到另一台机器。`model/` 内是真实基础模型权重（约 55.6 GB），不是下载占位文件、缓存软链接或 LoRA。数据、提示、分词器、解析代码和固定配置均已包含。

本包适配**一张完整 NVIDIA B200**，采用 **BF16、TP=1、总并发 16**。生成温度 **1.0**，`top_p=1.0`、`top_k=-1`，**thinking 开启**，每条最多生成 **16,384 tokens**（思考与最终答案合计），上下文上限 **32,768**。9,600 条源数据、顺序、攻击位置、配对、每条 seed、`user/input` 角色、提示文本、T+ q1 隔离定义都保持最新版本要求。

从原来“两组 TP=2、每组并发 8”适配到“一张 B200、TP=1、并发 16”。模型与采样参数未改变；更换硬件及并行方式可能导致浮点差异，因此不承诺与 A100 输出逐 token 相同。

## 推荐：使用固定 Docker 镜像

目标机器需要 Linux x86_64、Python 3.10+、Docker、NVIDIA Container Toolkit，以及支持 B200/CUDA 13 的 NVIDIA 驱动（建议 580 或更新；以实际预检为准）。不要求宿主机安装 PyTorch 或 CUDA Toolkit。GPU 应为空闲的完整 B200；脚本会拒绝 A100、多个 GPU 选择、较小 MIG 切片和已被占用的卡。

在这个文件夹中执行：

```bash
# 首次在目标机器上下载固定推理镜像；模型权重已经在包里。
./docker_b200.sh pull

# 检查 B200、驱动兼容性、单卡可见性和 BF16 运算。
./docker_b200.sh preflight --gpu 0

# 后台启动。退出 SSH 不会停止任务。
./docker_b200.sh start --gpu 0

# 查看进度；北京时间显示。
watch -t -n 5 ./docker_b200.sh status
```

`start` 自动完成模型及代码 SHA256 校验、固定环境检查、全部 9,600 条原生分词和请求一致性检查，然后加载模型。先生成同一组 **12 条**启动验证样本，要求原始 token 校验通过、自然 EOS、恰好一个 `</think>` 和非空最终答案；通过后自动继续其余 **9,588 条**。不会要求中途再确认。12 条计入总数 9,600，完成后自动退出模型服务并释放所属 GPU。

首次校验 55 GB 权重需要一些时间。`progress.json` 显示 `preflight` 时尚未提交生成请求；加载时显示 `model_loading`；验证时显示 `smoke`；全量时显示 `bulk`。

```bash
# 运行前只检查文件、Python 环境与全部提示，不申请 GPU。
./docker_b200.sh check

# 查看已落盘的主程序日志，Ctrl+C 只退出查看。
./docker_b200.sh logs

# 请求暂停：停止提交，等待已有请求保存完成，然后释放模型服务。
./docker_b200.sh pause

# 确认 status 为 paused、in flight 为 0 后，显式恢复同一任务。
./docker_b200.sh start --gpu 0 --resume
```

暂停可能需要等待长思考请求结束，单请求超时保持原版的 3,600 秒。全任务上限保持 48 小时。脚本只管理自己启动的进程/容器，不会结束其他任务。发生基础设施或解析错误时，保留完整错误类型、请求和已完成响应，停止新增请求，不自动重试、不把错误算成语义标签。`--resume` 只跳过已校验的完成记录；发现未完成或错误 case 时会拒绝静默重新生成，需先查看保存的错误。

如果要另开一套独立结果，可增加 `--run-id run2`；查看、暂停和恢复时也使用同一个 `--run-id`。默认 `main` 不覆盖已有数据。只选一张卡，例如 `--gpu 2` 或 `--gpu GPU-...`；不要传 `0,1`。

固定镜像：

```text
vllm/vllm-openai:v0.24.0@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f
```

其环境已检查为 vLLM 0.24.0、PyTorch 2.11.0+cu130、CUDA 13.0、Transformers 5.12.1，PyTorch 包含 `sm_100`。保留原来的 eager 执行，不自动开启量化、MTP、推测解码或新增 reasoning-effort 指令。镜像不包含在本文件夹中，目标机器需要首次 `pull` 或提前离线导入该固定镜像。

若在容器里调用外部 Docker daemon，且 daemon 看到的路径不同，可设置 `OPCD_DOCKER_BUNDLE=/实际宿主机上的本文件夹路径`。普通宿主机直接运行不需要此设置。不要混用 native 和 Docker 启停同一个 run ID。

## 无 Docker 的机器：原生方式

使用独立虚拟环境，不修改系统 Python：

```bash
./install_native.sh
./run_b200.sh preflight --gpu 0
./run_b200.sh start --gpu 0
watch -t -n 5 ./run_b200.sh status
```

`install_native.sh` 需要联网安装固定版本。可用 `OPCD_PYTHON=/已有环境/bin/python` 指定预装环境；预检会核对版本。原生方式通过独立后台进程运行，日志位于 `output/main/supervisor.log`。也支持 `check`、`pause`、`start --resume` 和 `--run-id`。对会在 SSH 退出时清理整个用户进程组的托管平台，应在平台提供的持久任务或 tmux 中运行 `./run_b200.sh run --gpu 0`。

## 结果在哪里

**全部任务产物统一放在本包的 `output/` 下。默认运行目录是 `output/main/`，Docker 和原生方式一致。运行结束后复制整个 `output/` 即可带走结果、主程序/服务日志、状态和运行回执。**

```text
output/
├── logs/                    # 启动、预检、检查、暂停和镜像下载命令日志
├── setup/                   # 原生依赖安装日志与 pip 缓存（使用原生安装时）
└── main/                    # 指定 --run-id run2 时为 run2/
    ├── supervisor.log       # 主程序 stdout/stderr 和异常堆栈，两种运行方式均保存
    ├── server.log           # vLLM 加载及推理日志
    ├── cases/               # 持续保存的每条请求、原始响应和轨迹
    ├── generations.jsonl    # 全部 9600 条完成后的汇总
    ├── progress.json       # 进度
    ├── cache/               # 框架运行缓存
    ├── tmp/                 # 运行临时文件
    └── *.json               # 配置、校验、汇总、暂停/失败/完成回执
```

Docker 的主程序日志直接写入宿主机挂载的 `output/main/supervisor.log`；不依赖容器日志存储，容器退出后文件仍在。`status` 和 `logs` 是只读查看命令，不会反复追加查看内容。

- `output/main/progress.json`：完成数、在途数、停止类型、生成 token 数、实测吞吐、ETA；完成至少 100 条后开始估 ETA。
- `output/main/cases/00000/`：每条的 `execution.json`、`request.json`、`raw_response.json`、`trajectory.json`。生成途中即可查看。
- `output/main/generations.jsonl`：9,600 条全部完成并通过最终原始 token/覆盖审计后生成的汇总文件。
- `output/main/smoke_gate.json`、`generation_summary.json`、`completion.json`：验证结果、全量汇总和退出回执。
- `output/main/server.log`：模型加载和推理服务日志。
- `output/main/hardware.json`、`environment.json`、`config.json`、`asset_gate.json`：目标硬件、环境、配置及文件校验依据。

这一步生成的是**训练候选轨迹**。每条仍标记 `training_ready=false`；A/U 标签、到达 token 上限样本的处理、T+ 真实模型前向、该模型的可靠性校准及训练集成仍须后续完成。包内不含 API 密钥，不自动调用付费标注，也不启动训练。

## 文件夹内容与验证边界

| 路径 | 内容 |
|---|---|
| `model/` | Qwen3.8-27B 基础模型：18 个权重分片及完整 tokenizer/config/license，共 32 个文件 |
| `inputs/` | 固定的 9,600 条 prepared 请求、T+ 视图与原版 12 条 canary 索引 |
| `source_data/` | 最新攻击源数据及配套 utility/dev 数据、来源 manifest |
| `code/` | 当前冻结版本的核心源码与原始 token 解析器 |
| `scripts/` | 可迁移的 B200 采集器和 Docker 启停工具 |
| `config/generation.json` | 单 B200 配置 |
| `provenance/` | 原版配置、摘要、验证回执和迁移差异；其中历史绝对路径仅作来源记录 |
| `validation/` | 打包时的本地检查记录，不能作为 B200 完整执行成绩 |
| `bundle_manifest.json` | 所有静态文件的相对路径、大小及 SHA256 |
| `output/` | 目标机器的新结果；交付时为空 |

基础模型固定为 `Qwen/Qwen3.8-27B`，revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。复制到新机器后可先运行 `python3 scripts/b200_collection.py verify` 完整校验，或 `verify --quick` 只跳过大权重的内容哈希（检查大小；不等于完整校验）。正式 `start` 始终执行完整校验。

本地验证在 A100 主机上进行，B200 硬件执行尚未发生。已适配代码与实际在 B200 跑通是两件事；目标机必须通过预检和 12 条真实生成验证才进入全量。

官方环境说明：[vLLM GPU 安装文档](https://docs.vllm.ai/en/v0.21.0/getting_started/installation/gpu/)说明 B200 至少需要 CUDA 12.8；本包固定采用已含 `sm_100` 的 CUDA 13.0 镜像。模型来源：[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)。
