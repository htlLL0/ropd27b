# OPCD：Qwen3.8-27B 单张 B200 数据生成

在一张 NVIDIA B200 上，使用固定的 Qwen3.8-27B 基础模型生成 9,600 条训练候选轨迹。运行配置为 BF16、TP=1、总并发 16、温度 1.0、thinking 开启，生成上限为每条 16,384 tokens。

- **操作步骤：[使用说明.md](使用说明.md)**
- 配置和技术交接：[README_先看这里.md](README_先看这里.md)
- 固定配置：[config/generation.json](config/generation.json)

## Git 仓库与完整运行包

Git 仓库保存代码、配置、说明、固定索引和来源清单。**模型权重、较大的 JSONL 数据、运行输出和本地环境已通过 `.gitignore` 排除，但保留在本地完整运行包中。**

从 GitHub 克隆代码后，需要从对应版本的完整运行包补齐下列文件，才能执行数据生成：

```text
model/                                  # 完整基础模型及 tokenizer/config，共 32 个文件
inputs/prepared.jsonl                    # 9600 条固定请求
inputs/teacher_views.jsonl               # 对应的 T+ 视图
source_data/attack_candidates.jsonl
source_data/parent_utility_controls.jsonl
source_data/parent_dev_attack_candidates.jsonl
```

使用与 `bundle_manifest.json` 对应的原始文件；不要重新抽样或替换为其他版本。`inputs/smoke_selection.json`、`source_data/manifest.json` 和来源记录随代码保留。

模型固定为 `Qwen/Qwen3.8-27B`，revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。Git 仓库不保存 55.6 GB 的权重，也未配置 Git LFS。

## 启动

补齐上述资源后，进入仓库目录执行：

```bash
./docker_b200.sh pull
./docker_b200.sh preflight --gpu 0
./docker_b200.sh start --gpu 0
```

目标机器需要支持 GPU 的 Docker 环境与空闲的完整 B200。原生安装方式、环境要求、暂停恢复及结果检查见 [使用说明](使用说明.md)。

所有任务结果和日志统一保存到 `output/`，默认任务目录为 `output/main/`。模型、输入、运行输出及 `.venv/` 都不应通过普通 `git add` 纳入提交。

## 验证范围

本包已完成本地代码、模型文件及 9,600 条输入检查，并验证日志可在容器退出后保留。B200 上的实际执行仍需目标机预检和 12 条真实生成验证；通过后自动继续全量。

本任务仅生成训练候选轨迹，不启动训练或付费标注。基础模型许可保存在完整模型目录的 `model/LICENSE`。
