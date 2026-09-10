# Qwen3.8-27B 切换交接

用户于 2026-09-10 要求：保存当前结果，改用 Qwen3.8-27B，其余遵循中午确认的最新版本。本文件记录已经执行的切换；运行计数以新目录的 `progress.json` 为准。

## 已保存的旧版

- 旧目录：`artifacts/secopd_qwen36_27b_9600_roles_think_t1_16k_20260910`。
- 完成输出 698 条，其中自然 EOS 697 条、达到 16,384 tokens 上限 1 条（索引 252）；均保留，未重采样。
- 原始请求、完整响应、实际 token IDs、解析结果、执行记录全部原位保留，698 条原始输出审计通过。
- 额外汇总：`generations.partial.model_switch.jsonl`，SHA256 为 `72fd260d11022ae1c63bcd14306b812bdfe8f34dd068c4a31fa5b580b26145f2`。
- 归档及逐文件校验：`MODEL_SWITCH_ARCHIVE_20260910.json`。
- SIGUSR1 停止新增请求后，等待中的最后两条长请求仍未结束。为响应尽快切换的要求，仅停止已核实属于本任务的两个 worker，索引 650、717 的请求保留并写入 `model_switch_interruption.json`；它们不是完成输出，不计入 698 条。
- 旧 supervisor 已退出，所属 GPU server 已停止。`failure.json` 的 `RuntimeError: User requested pause` 是本次用户暂停的预期控制流程。
- 旧 `progress.json` 保留 worker 停止前的 `in_flight=2` 历史计数；归档中 `active_requests=0` 和容器终态给出最终状态。不得据此误判仍有两个运行请求。
- 旧模型任务没有自动恢复授权。

## 已启动的新任务

- 目录：`artifacts/secopd_qwen38_27b_9600_roles_think_t1_16k_20260910`。
- 模型：`Qwen/Qwen3.8-27B`，固定 revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。
- 全部 32 个模型文件已对照官方 blob/LFS 摘要校验。模型以本地只读路径挂载，不修改共享缓存。
- supervisor PID：1237272，仅为启动时记录；进程是否仍存活需实时核对。
- GPU 2/5 和 3/4，各 TP=2、BF16、并发 8，总并发 16。GPU 0/1 的其他任务未操作。
- 53 项 CPU 测试通过；9,600 条 Student/T+ 输入逐条核对通过。源行、顺序、攻击位置、配对、seed、输入文本和 token IDs 均与旧版一致。
- 新旧配置差异仅为模型身份、路径、revision、manifest、served name，以及 schema/创建时间元数据。核对凭据：`model_only_diff.json`。
- temperature=1.0、top_p=1.0、top_k=-1、min_p=0，thinking 开启，最大生成 16,384 tokens（含思考和最终答案），上下文上限 32,768；惩罚项与最新旧版相同。
- 保留可信任务 `user` 与被攻击外部内容 `input` 的分离、原始提示渲染、T+ q1 注意力隔离定义和所有现有训练方法要求。
- 无新增 reasoning-effort 指令、量化、MTP、推测解码或采样默认值。
- 两路先生成原来同一组 12 条 canary，实际 token、单一思考结束边界、非空最终答案、自然 EOS 全部通过后自动继续全量。12 条计入 9,600。
- 新模型从头生成自己的 9,600 条；旧模型输出和旧 A/U 标签不混入。
- 完整覆盖审计及正常完成后释放所属 GPU 的机制不变。未开始训练或新的付费标注。

独立的 temperature=0 API 标注任务仍在运行，余额不足的 group_b 仍保持 hold；本次切换未操作该任务。

## 查看进度

```bash
cd "/home/shuai/AAAI 2027/safety_opcd_plan/0820/R-OPCD-V3/artifacts/secopd_qwen38_27b_9600_roles_think_t1_16k_20260910"
watch -t -n 2 -x python3 show_progress.py
```

查看原始 `progress.json`、`smoke_gate.json`、`failure.json` 和 `receipts/` 可以区分加载、小批量验证、全量生成及异常。模型替换本身不保证速度提升。

## 单张 B200 的询价估算

这是估算，不是迁移授权，也不是目标硬件实测。单张 B200、BF16、总并发 16，保留温度和思考设置，假设新模型平均输出接近旧版：旧版 698 条共 1,268,605 个生成 tokens，均值 1,817.49，外推 9,600 条约 1,744.8 万 tokens。

以有效总输出吞吐 500–900 tokens/s 为规划假设，纯生成约 5.4–9.7 小时，因此粗估 6–10 小时、租机先预留 12 小时；模型下载和环境安装另计。若 Qwen3.8 思考长度增加，耗时近似随总输出 token 数增加。应以 B200 上代表性小批量重新校准。

参考仅用于确定量级，不直接当成本任务实测：
- NVIDIA B200 180 GB、8 TB/s 规格：https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html
- Qwen3.5-27B 单 B200 BF16 自定义实现，batch 16 报告约 724 tokens/s：https://github.com/RightNow-AI/qwen3.5-triton
- Qwen3.5-27B 单 B200 SGLang 自回归基线，并发 16 报告约 1,031–1,121 tokens/s，使用的是 AR 列，不采用 MTP/DFlash 加速列：https://huggingface.co/z-lab/Qwen3.5-27B-DFlash/blob/8f5c0a6d736cfdc2967229e1dd3d3e9f15624d8b/README.md

当前采用 vLLM eager，以上引擎、模型和工作负载均有差异，500–900 tokens/s 仍需目标硬件验证。
