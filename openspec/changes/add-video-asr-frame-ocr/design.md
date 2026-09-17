## Context

现有 FastAPI 服务只接收图片并直接调用 pytesseract。参考 ASR 项目定义了以毫秒为单位的 `segments` 和 OCR `detections` 交换格式；视频处理必须保留临时文件边界，避免将上传内容全部加载到内存或覆盖原始附件。

## Goals / Non-Goals

**Goals:**

- 增加一个清晰的 multipart 视频 OCR 接口，兼容参考项目的回传字段。
- 用 FFmpeg 的时间选择与帧输出取得可被 PIL/Tesseract 处理的图片，并保留实际 PTS。
- 让抽帧逻辑可单元测试，验证窗口、去重、排序和输入校验。

**Non-Goals:**

- 不修改 ASR transcript，不自动提出或应用同音字纠错。
- 不实现字幕区域检测、复杂字幕跟踪或人工审核工作流。
- 不移除或改变既有图片 `/ocr` 契约。

## Decisions

1. **接口形式**：新增 `POST /ocr/video`，字段为 `video` 与 `transcript`，可选 `sample_interval_ms` 不暴露为首版配置，固定 1000ms 以保持契约简单。备选是将视频 URL 放入 JSON，但会引入远程访问和安全边界。
2. **FFmpeg 集成**：通过 `subprocess.run` 调用系统 `ffmpeg`，使用 `-ss`/`-frames:v 1` 为每个目标点取帧，并以 `-vf showinfo` 读取解码帧 PTS；若 showinfo 不可用则使用请求点作为受控回退并在实现中保持毫秒单位。备选是 OpenCV，增加较重依赖且时间戳语义不够稳定。
3. **抽帧策略**：每个 segment 取 `[start_ms-500, end_ms+500]` 的边界内点，至少取中点/起点之一，长窗每 1000ms 一个点，合并相邻重复点。该策略覆盖短暂字幕，同时控制 OCR 成本。
4. **OCR 结果**：使用 pytesseract 的 `image_to_data` 获取文字、置信度和词框，合并同一帧的有效文本为一条 subtitle detection；词框归一化后输出。不能可靠估算模型 revision 时使用当前包版本字符串。
5. **资源与安全**：上传内容写入 `TemporaryDirectory`，请求结束自动清理；限制 transcript 大小和 segment 数量，视频由 FFmpeg 读取本地临时路径，不接受客户端命令参数。

## Risks / Trade-offs

- [FFmpeg 未安装或版本差异] -> Dockerfile 显式安装 FFmpeg，并在服务启动/请求错误中给出可诊断信息。
- [逐点启动 FFmpeg 成本较高] -> 先采用易验证的逐点抽帧实现；未来可按连续窗口批量解码优化。
- [字幕区域和水印难以区分] -> 首版只输出 `kind=subtitle` 的 OCR 结果，明确由下游或人工审核处理误检。

## Migration Plan

部署新镜像并确认 `/health`；旧客户端继续使用 `/ocr`。回滚时恢复旧镜像即可，新端点不改变既有数据。
