## Purpose

提供面向短剧质检的时间轴驱动视频字幕 OCR，使 ASR 片段、解码帧和 OCR 结果能够在统一的视频毫秒时间基准下关联。

## ADDED Requirements

### Requirement: Video OCR SHALL accept ASR-timed input

系统 SHALL 提供视频 OCR 接口，接受一个视频文件和 UTF-8 JSON transcript；transcript SHALL 至少包含 `segments` 数组，数组元素 SHALL 包含非负的 `start_ms` 与大于 `start_ms` 的 `end_ms`。系统 SHALL 拒绝无法解析的 JSON、缺少必要字段或时间区间非法的请求，并返回可诊断的 4xx 错误。

#### Scenario: Valid transcript produces a timed OCR document
- **WHEN** 客户端上传可解码视频和包含有效 `segments` 的 transcript JSON
- **THEN** 系统返回 HTTP 200，并返回 `media_id`、`timebase: "video_ms"` 和 `detections` 数组

#### Scenario: Invalid transcript is rejected
- **WHEN** transcript 不是 JSON、缺少 `segments` 或任一片段时间无效
- **THEN** 系统返回 HTTP 400，且不执行 OCR

### Requirement: Frame sampling SHALL follow ASR windows

系统 SHALL 以每条 ASR 片段为抽帧窗口，将窗口前后扩展最多 500ms，并在窗口内按约 1000ms 间隔生成抽帧点；系统 SHALL 合并重复抽帧点并按时间升序处理。抽帧点 SHALL 使用视频解码得到的真实帧 PTS，而非仅依赖请求时间。

#### Scenario: Short and long segments receive coverage
- **WHEN** transcript 同时包含短片段和超过一秒的长片段
- **THEN** 每个片段扩展窗口至少得到一个帧，长片段得到约每秒一个帧，且不会因相邻片段重叠而重复处理同一时间点

#### Scenario: Frame decode failure is reported
- **WHEN** 视频不存在、格式不可解码或 FFmpeg 无法取得任何请求帧
- **THEN** 系统返回 HTTP 422 或 500 的明确错误，而不是返回无时间信息的 OCR 结果

### Requirement: OCR output SHALL remain contract-compatible

系统 SHALL 返回 `media_id`（视频文件 SHA-256）、`timebase`、以及每条包含唯一 `id`、`frame_pts_ms`、归一化 `[x,y,width,height]` `bbox`、`ocr_text`、`confidence`、`engine_version`、`kind` 和 `asr_text_prompted` 字段的 detection。无法检测字幕的帧 SHALL 不生成空 detection；`kind` SHALL 为 `subtitle`。

#### Scenario: Detection fields can be consumed by ASR OCR tools
- **WHEN** 帧中识别出字幕文本
- **THEN** detection 的 `frame_pts_ms` 使用视频毫秒时间基准，bbox 每个值在 0 到 1 之间，且 `asr_text_prompted` 为 false

#### Scenario: Existing image OCR remains compatible
- **WHEN** 客户端继续向 `/ocr` 上传 image/* 文件
- **THEN** 系统继续返回原有的 `filename` 与 `text` 字段，不要求 transcript 或 FFmpeg
