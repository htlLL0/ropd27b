# 在另一台 B200 服务器上生成 9,600 条数据

按本 README 操作即可。**仓库已包含全部代码、输入数据和模型配套文件，只需另行补入 18 个模型权重分片。全部日志和结果统一保存在仓库的 `output/` 中。**

固定配置：Qwen3.8-27B、单张 B200、BF16、TP=1、并发 16、温度 1.0、`top_p=1.0`、thinking 开启，每条最多生成 16,384 tokens，上下文上限 32,768。无需修改配置或重新生成输入数据。

## 先确认服务器环境

推荐使用 Docker。目标服务器需要：

- Linux x86_64、Git、Python 3.10 或更新。
- 一张完整、空闲的 NVIDIA B200，NVIDIA 驱动支持 CUDA 13（建议 580 或更新，以预检为准）。
- Docker 和已配置的 NVIDIA Container Toolkit，当前账号能够运行 GPU 容器。
- 足够的磁盘空间：模型及数据约 55.84 GB，另外为 Docker 镜像、生成结果和缓存预留空间。
- 首次能够访问 GitHub 和 Docker 镜像仓库；权重、依赖和镜像准备好后，生成过程不需要联网。

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

### 2. 把权重放入 model/

使用 **`Qwen/Qwen3.8-27B` 的 revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`**。将对应的 18 个 `model-*-of-*.safetensors` 文件上传到 B200 服务器的 `$HOME/ropd27b/model/`，总计约 55.6 GB。

如果权重已经位于 B200 服务器的其他目录，替换源路径后复制：

```bash
cp /path/to/Qwen3.8-27B/model-*-of-*.safetensors ./model/
```

如果要从原服务器拉取，在 B200 服务器上执行以下命令，将账号、地址和源目录替换为实际值（需要两端均有 `rsync`，且 SSH 可以连接）：

```bash
rsync -avP -s --include='*.safetensors' --exclude='*' \
  'SOURCE_USER@SOURCE_HOST:/absolute/path/to/model/' ./model/
```

两种方式选一种即可。复制真实权重文件，不要放软链接或 Git LFS 指针；保留仓库中已有的 tokenizer、配置及权重索引。

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
├── logs/                    # 镜像下载、预检、启动、暂停等命令日志
├── setup/                   # 原生安装日志及 pip 缓存（仅原生方式）
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

完成后复制整个 `output/` 即可带走日志、结果和回执。生成的是训练候选轨迹，`training_ready=false` 为预期状态，后续仍需标注和训练集成；本任务不会自动启动训练。

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

先完成前面的仓库克隆和权重复制。使用带 `venv` 模块的 Python 3.12，安装依赖需要联网；驱动和 B200 的要求相同。

```bash
./install_native.sh
./run_b200.sh preflight --gpu 0
# 预检通过后启动
./run_b200.sh start --gpu 0
watch -t -n 5 ./run_b200.sh status
```

依赖安装在仓库的 `.venv/`，安装日志为 `output/setup/install_native.log`，其余结果布局与 Docker 相同。

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
| 找不到 `nvidia` runtime 或 GPU | 检查 NVIDIA Container Toolkit 是否已为 Docker 配置 |
| `B200 required` / GPU 已被占用 | 核对 `nvidia-smi`，选择正确且空闲的完整 B200 |
| 文件缺失、软链接、大小或 SHA256 不匹配 | 检查 18 个权重是否传输完整、revision 是否正确，以及是否混用了不同版本文件 |
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

本项目的操作说明统一在本 README。模型卡保存在 [model/MODEL_CARD.md](model/MODEL_CARD.md)，许可见 [model/LICENSE](model/LICENSE)。`provenance/` 和 `validation/` 保留历史来源及验证记录，其中旧文件名或摘要对应当时的版本。
