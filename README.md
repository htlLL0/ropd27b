# OPCD：Qwen3.8-27B 单张 B200 数据生成

在一张 NVIDIA B200 上，使用固定的 Qwen3.8-27B 基础模型生成 9,600 条训练候选轨迹。运行配置为 BF16、TP=1、总并发 16、温度 1.0、thinking 开启，生成上限为每条 16,384 tokens。

- **操作步骤：[使用说明.md](使用说明.md)**
- 配置和技术交接：[README_先看这里.md](README_先看这里.md)
- 固定配置：[config/generation.json](config/generation.json)

## Git 仓库与模型权重

**本运行包除 18 个模型权重分片外，所有文件均随 Git 仓库提供。** 包括固定的 9,600 条输入、T+ 视图、全部源数据、完整 tokenizer/config、权重索引、模型许可、代码、配置、说明和本地验证记录。数据和 tokenizer 直接存入 Git，无需 Git LFS。

从 GitHub 克隆后，只需向 `model/` 补入同一版本的 18 个 `model-*-of-*.safetensors` 权重分片（约 55.6 GB）。不要替换仓库里的 tokenizer、配置、输入或源数据。

模型固定为 `Qwen/Qwen3.8-27B`，revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。权重可以从现有完整运行包复制；文件大小和 SHA256 以 `bundle_manifest.json` 为准。

## 启动

首次从 GitHub 获取并补入权重（将复制命令中的路径替换为实际模型目录）：

```bash
git clone https://github.com/htlLL0/ropd27b.git OPCD_Qwen38_27B_B200_9600_20260910
cd OPCD_Qwen38_27B_B200_9600_20260910
cp /path/to/Qwen3.8-27B/model-*-of-*.safetensors model/
python3 scripts/b200_collection.py verify
```

完整复制本地运行包时，权重已经在 `model/`，无需再次补入。然后执行：

```bash
./docker_b200.sh pull
./docker_b200.sh preflight --gpu 0
./docker_b200.sh start --gpu 0
```

目标机器需要支持 GPU 的 Docker 环境与空闲的完整 B200。原生安装方式、环境要求、暂停恢复及结果检查见 [使用说明](使用说明.md)。

所有任务结果和日志统一保存到 `output/`，默认任务目录为 `output/main/`。交付时该目录为空；Git 不保存空目录，启动脚本会自动创建。虚拟环境、缓存和凭据仍按常规规则忽略，包内没有这些待上传文件。

## 验证范围

本包已完成本地代码、模型文件及 9,600 条输入检查，并验证日志可在容器退出后保留。B200 上的实际执行仍需目标机预检和 12 条真实生成验证；通过后自动继续全量。

`validation/` 保留各次本地检查的历史记录，其中旧提交或旧清单摘要对应当时的快照。为直接分发 tokenizer，`model/.gitattributes` 仅追加了取消 tokenizer 的 LFS 过滤规则；模型配置、tokenizer 内容及权重未修改。

本任务仅生成训练候选轨迹，不启动训练或付费标注。基础模型许可见 [model/LICENSE](model/LICENSE)。
