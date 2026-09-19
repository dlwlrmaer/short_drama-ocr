# Proposal

## Why

当前服务仍使用 Tesseract 和稀疏的逐帧输出，无法复用最新 Windows 11 实测中已经验证的 RapidOCR PP-OCRv6 small、字幕区域过滤和短字幕边界选帧策略，也没有使用本机 RTX 4070。需要把评估脚本中的冻结参数转成可测试、可配置且能在 Win11 主机优先通过 NVIDIA GPU 运行的服务能力，并明确区分已验证的 OCR 选择逻辑与仍待真实 transcript 验证的 ASR 增益。

## What Changes

- 增加 Win11 原生安装、模型准备、启动和健康检查流程，不依赖 Docker 或本机 Tesseract，并支持 `auto`、`cuda`、`cpu` 三种执行模式。
- 将默认 OCR 后端切换为 RapidOCR PP-OCRv6 small；`auto` 模式在 NVIDIA CUDA Provider 可用时优先使用 GPU，不可用时记录原因并回退 CPU，同时保持现有 `/ocr` 图片响应字段兼容。
- 将 `/ocr/video` 升级为候选帧、字幕区域识别、文字框过滤和按 segment 选帧的流水线：保留全片 1000ms 网格，在 ASR 时间窗前后各扩 500ms 并以 250ms 加密采样。
- 固化最新评估默认值：ROI `[0.10, 0.44, 0.85, 0.82]`、全画面纵向中心 `0.715–0.815`、首尾各裁剪 100ms、解码 PTS 保护 40ms、至少 3 个内部候选、单字 100ms 双帧确认，以及单帧和未裁剪回退。
- 在不使用 ASR 文本作为 OCR 真值或提示词的前提下，为每个 segment 返回所选真实 PTS、归一化框、文字、置信度和选择证据；保留现有 detection 基础字段。
- 把评估脚本中的通用候选生成、框轨迹和选择逻辑提取为生产模块，并增加单元、API、FFmpeg、CUDA Provider 冒烟和本机冻结样本回归验证。
- 增加同一批样本的 GPU/CPU 质量与性能对照，报告 OCR 吞吐量、预热后 P50/P95 延迟、端到端耗时和实际 Provider；只有 GPU 保持质量且实际提速时才将其作为本机推荐路径。
- 更新互相冲突的模型选择文档，以最新《瑶瑶如她》评估结论作为 Win11 默认方案，并补充 GPU 实测及完整评估集 ASR 时间窗对照结果。
- 增加 ASR 文件适配层和 Win11 命令行入口，直接读取 `transcript.json` 的 `source_file`、`media_id` 与时间段，校验源视频身份后原子写出可供 `ocr-propose` 使用的 `ocr_evidence.json`。
- 增加跨主机运行 profile：Windows 11 自动使用 PP-OCRv6 small 并优先 CUDA；Linux 低显存主机自动使用 PP-OCRv5 mobile 与 CPU，允许用环境变量显式覆盖模型和 Provider。
- 图片 OCR 同时支持单图和有上限的批量上传；完整视频帧可启用字幕 ROI 与背景硬遮罩，批量上限和遮罩分支按检测到的系统内存与显存自动调整。

## Capabilities

### New Capabilities

- `win11-rapidocr-subtitle-pipeline`: 在 Windows 11 主机上优先使用 NVIDIA CUDA、必要时回退 CPU 来运行 RapidOCR 字幕服务，按冻结的候选、区域过滤和边界选择规则输出与现有视频 OCR 契约兼容的 segment 级字幕检测结果。

### Modified Capabilities

无。当前主规范目录没有已归档能力；本变更建立在尚未归档的 `add-video-asr-frame-ocr` 实现之上，但不把不存在的主规范声明为修改项。

## Impact

- 影响 FastAPI OCR 后端、视频抽帧与选择模块、依赖清单、Windows PowerShell 启动工具、测试和 README/评估文档。
- `/ocr/video` 仍接受 `video` 与 `transcript`，返回的 detection 保留 `id`、`frame_pts_ms`、`bbox`、`ocr_text`、`confidence`、`engine_version`、`kind`、`asr_text_prompted`；新增 segment 关联和选择证据字段。
- 本地 ASR 流程可直接调用 `app.asr_adapter` 或 `scripts/ocr_from_asr.py`，不再需要复制或重新上传源视频；文件适配器在推理前校验 ASR `media_id` 与源视频 SHA-256。
- 默认运行时新增 RapidOCR 与 ONNX Runtime GPU/CPU 二选一安装路径，Win11 主机需提供 Python 3.11 和 FFmpeg；CUDA 路径还需兼容的 NVIDIA 驱动与 CUDA/cuDNN 运行库。Linux 低显存 profile 默认安装和使用 CPU Runtime，不要求 Docker、WSL、Ollama 或 Tesseract。
- 健康检查和运行日志新增请求 Provider、实际 Provider、GPU 设备与回退原因；显式 `cuda` 模式不得静默回退。
- 本轮验收以 Win11 RTX 4070 与 CPU 的同样本对照、本地冻结样本和合成测试为主；NAS/Linux 性能与真实 ASR 引导增益不作为本变更的完成条件。
