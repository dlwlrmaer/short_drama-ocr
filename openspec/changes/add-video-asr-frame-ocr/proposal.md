## Why

当前 API 只能处理单张图片，无法直接利用短剧 ASR 时间轴从视频取得字幕画面，导致 OCR 流程需要手工抽帧且难以与 ASR 结果对齐。新增视频抽帧 OCR 能力可让短剧字幕质检使用统一的时间坐标和回传格式。

## What Changes

- 新增接受视频与 ASR transcript JSON 的视频 OCR 接口。
- 按每条 ASR 片段扩展前后 500ms、约每秒抽取一帧，并以视频时间轴返回 OCR 检测结果。
- 使用 FFmpeg 解码视频真实帧，保留现有图片 `/ocr` 接口兼容性。
- 增加输入校验、媒体哈希、FFmpeg 错误处理、测试和运行文档。

## Capabilities

### New Capabilities

- `video-asr-frame-ocr`: 根据 ASR 时间轴从视频抽帧并执行字幕 OCR，输出可供 ASR 下游消费的检测结果。

### Modified Capabilities

无。

## Impact

- 影响 `app/main.py`、依赖清单、Docker 镜像和 README。
- 运行环境需要提供 FFmpeg/ffprobe 命令。
- 新接口新增 multipart 视频与 JSON 文件上传，但现有图片 OCR API 不变。
