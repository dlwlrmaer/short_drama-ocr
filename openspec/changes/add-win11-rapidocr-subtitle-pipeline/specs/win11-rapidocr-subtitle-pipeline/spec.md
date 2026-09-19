# Spec Delta

## Purpose

提供可在 Windows 11 主机直接安装和运行、优先利用 NVIDIA GPU 并可回退 CPU 的短剧字幕 OCR 服务，将已冻结的候选帧、字幕区域、文字框过滤与边界选帧规则用于图片和视频接口，同时保持既有响应契约可被下游继续消费。

## ADDED Requirements

### Requirement: Runtime SHALL select a host profile with explicit overrides

系统 SHALL 支持 `auto`、`win11`、`linux` 与 `linux-low-vram` 运行 profile。`auto` SHALL 检测操作系统、系统内存、NVIDIA GPU 与显存：Windows 解析为 `win11`；Linux 在 NVIDIA 显存不少于 4GB 时解析为 `linux`，否则解析为 `linux-low-vram`。`win11/linux` SHALL 使用 PP-OCRv6 small、CUDA 优先和自适应背景抑制；`linux-low-vram` SHALL 使用 PP-OCRv5 mobile、CPU 和单分支空间遮罩。`OCR_EXECUTION_MODE`、`OCR_MODEL_PROFILE`、`OCR_BACKGROUND_SUPPRESSION`、`OCR_MAX_BATCH_IMAGES` 与 `OCR_GPU_DEVICE_ID` SHALL 能覆盖自动默认值。健康检查 SHALL 返回请求 profile、实际 profile、硬件分级、检测到的系统/显存、批量上限、背景抑制、模型和实际 Provider。

#### Scenario: Auto profile runs on Windows 11
- **WHEN** 用户未显式选择 profile 且在 Windows 11 启动服务
- **THEN** 系统选择 `win11`、加载 PP-OCRv6 small，并按 `auto` 执行模式优先尝试 CUDA

#### Scenario: Auto profile protects a Linux host with 2GB VRAM
- **WHEN** 用户未显式选择 profile 且在 Linux 启动服务
- **AND** 检测到 NVIDIA 显存为 2GB
- **THEN** 系统选择 `linux-low-vram`、加载 PP-OCRv5 mobile，并只创建 CPU Provider 会话

#### Scenario: Auto profile uses a capable Linux GPU
- **WHEN** 用户未显式选择 profile且 Linux 主机的 NVIDIA 显存不少于 4GB
- **THEN** 系统选择 `linux`、加载 PP-OCRv6 small，并以 `auto` 执行模式尝试 CUDA

#### Scenario: Explicit settings override profile defaults
- **WHEN** 用户设置有效的执行模式或模型环境变量
- **THEN** 系统使用显式值，并在健康检查中报告最终模型和 Provider

### Requirement: Win11 host SHALL support explicit GPU and CPU execution modes

系统 SHALL 提供从 Windows 11、Python 3.11 虚拟环境安装依赖并启动服务的受支持流程，并 SHALL 支持 `auto`、`cuda` 与 `cpu` 三种执行模式。Win11 profile 运行时 SHALL 使用 RapidOCR PP-OCRv6 small；`auto` 模式 SHALL 在 `CUDAExecutionProvider` 可创建实际推理会话时优先使用 NVIDIA GPU，否则记录原因并回退 `CPUExecutionProvider`；`cuda` 模式 SHALL 在 CUDA Provider 不可用时使 OCR 就绪检查失败且 SHALL NOT 静默回退；`cpu` 模式 SHALL 只使用 CPU Provider。系统 SHALL NOT 要求 Docker、WSL、Ollama 或系统 Tesseract。

#### Scenario: Auto mode selects the available NVIDIA GPU
- **WHEN** 用户在具有兼容 NVIDIA 驱动、CUDA 运行库和 CUDA Provider 的 Win11 主机以 `auto` 模式启动服务
- **THEN** 服务使用 `CUDAExecutionProvider` 完成预热，健康检查返回 PP-OCRv6 small、GPU 设备和实际 CUDA Provider

#### Scenario: Auto mode falls back to CPU with a reason
- **WHEN** 用户以 `auto` 模式启动服务但 CUDA Provider 无法创建实际推理会话
- **THEN** 服务使用 `CPUExecutionProvider` 完成预热，并在健康检查和日志中返回 CUDA 不可用的具体原因

#### Scenario: Explicit CUDA mode cannot silently fall back
- **WHEN** 用户以 `cuda` 模式启动服务但 CUDA Provider、兼容 DLL 或 GPU 设备不可用
- **THEN** OCR 就绪检查失败并返回可诊断错误，系统 SHALL NOT 将 CPU 推理报告为 CUDA 推理

#### Scenario: Explicit CPU mode remains supported
- **WHEN** 用户以 `cpu` 模式启动服务
- **THEN** 服务只创建 CPU Provider 会话，并继续支持图片和视频 OCR

#### Scenario: Required executable is unavailable
- **WHEN** Win11 主机找不到 FFmpeg 或 OCR 引擎无法初始化
- **THEN** 启动或就绪检查返回明确的缺失组件与修复提示，视频 OCR SHALL NOT 被报告为可用

### Requirement: Runtime status SHALL expose the actual execution provider

系统 SHALL 在健康检查和性能报告中公开请求的执行模式、实际 ONNX Runtime Provider、GPU 设备标识、模型标识与回退原因。系统 SHALL 以实际推理会话使用的 Provider 为准，不能只根据驱动存在、配置值或包名宣称 GPU 已启用。

#### Scenario: GPU package is installed but session uses CPU
- **WHEN** GPU 相关包或驱动存在，但 PP-OCRv6 small 的实际推理会话仅使用 CPU Provider
- **THEN** 状态 SHALL 报告实际 Provider 为 CPU，并附带 GPU 选择失败或回退原因

#### Scenario: CUDA session is active
- **WHEN** PP-OCRv6 small 的检测、方向分类与识别会话成功使用 CUDA Provider
- **THEN** 状态 SHALL 报告 CUDA Provider 与 GPU 设备，且后续性能报告 SHALL 使用相同 Provider 标识

### Requirement: Image OCR SHALL preserve its public response contract

系统 SHALL 使用默认 OCR 引擎处理 `POST /ocr` 的有效图片，并继续返回 `filename` 与 `text` 字段。系统 SHALL 保留现有图片类型和解码错误的 4xx 行为。

#### Scenario: Existing image client remains compatible
- **WHEN** 客户端以 `image/*` 上传可解码图片到 `/ocr`
- **THEN** 系统返回 HTTP 200，响应仍包含原文件名和去除首尾空白的识别文本

#### Scenario: Invalid image is rejected
- **WHEN** 上传内容不是图片类型或无法解码为图片
- **THEN** 系统返回可诊断的 HTTP 4xx 响应且不执行 OCR 推理

### Requirement: Image OCR SHALL accept single and bounded batch inputs

系统 SHALL 保留 `POST /ocr` 单图接口，并 SHALL 提供 `POST /ocr/batch` 多图接口。批量接口 SHALL 保持上传顺序，逐张解码和推理，共用进程级 Provider，并 SHALL 返回包含 `index`、`filename`、`text` 的结果数组。系统 SHALL 根据硬件 profile 限制单批图片数，且 SHALL NOT 同时保留整批未压缩图片。

#### Scenario: Batch preserves order and provider reuse
- **WHEN** 客户端上传多张有效图片
- **THEN** 系统按输入顺序返回全部结果，并只复用同一个已就绪 OCR Provider

#### Scenario: Batch exceeds the active hardware limit
- **WHEN** 上传图片数量超过当前 profile 的批量上限
- **THEN** 系统在解码图片前返回 HTTP 413 和实际批量上限

#### Scenario: Full-frame subtitle input enables suppression
- **WHEN** 单图或批量图片请求设置 `subtitle=true`
- **THEN** 系统在 OCR 前应用字幕 ROI 与 profile 对应的背景遮罩，并在 OCR 后继续应用字幕框中心过滤

### Requirement: Video candidate schedule SHALL combine global fallback and timed dense sampling

系统 SHALL 根据视频时长生成包含首尾边界的 1000ms 全片网格，并为 transcript 中每个有效 segment 生成前后各扩展 500ms、间隔 250ms 的加密候选；所有请求点 SHALL 限制在视频范围内、去重并排序。系统 SHALL 使用解码得到的真实帧 PTS，并 SHALL 将与请求点相差超过 40ms 的帧标记为不可用于内部边界判定。

#### Scenario: Dense schedule augments the fallback grid
- **WHEN** 视频包含不与 1000ms 网格对齐的短 segment
- **THEN** 候选集合包含完整的全片网格，并包含该 segment 扩展窗口中的 250ms 采样点且没有重复点

#### Scenario: Decode timestamp exceeds the guard
- **WHEN** 解码帧真实 PTS 与用于某个内部边界锚点的请求时间相差超过 40ms
- **THEN** 系统保留真实 PTS 供诊断，但不把该帧计作满足该边界锚点的候选

### Requirement: Subtitle OCR SHALL use the frozen region and box filter defaults

系统 SHALL 默认只识别归一化全画面区域 `[0.10, 0.44, 0.85, 0.82]`，在 OCR 前 SHALL 能将字幕中心带及其安全边距之外的 ROI 像素涂黑，并 SHALL 只把换算到全画面后纵向中心位于 `0.715–0.815` 的文字框用于字幕文本、置信度与合并框。自适应模式 SHALL 比较原始 ROI 与空间遮罩 ROI，低资源模式 SHALL 只运行空间遮罩分支。返回的 `bbox` SHALL 使用全画面的归一化 `[x,y,width,height]` 坐标并将各值限制在 0 到 1。

#### Scenario: Background text is outside the subtitle band
- **WHEN** OCR 同时检测到字幕带内文字框和纵向中心位于过滤带外的标识或道具文字框
- **THEN** 候选字幕文本、置信度和合并框只由字幕带内文字框组成

#### Scenario: No box survives filtering
- **WHEN** 某候选帧的全部文字框均位于字幕过滤带外
- **THEN** 该帧被视为没有字幕文本且不会单独生成 detection

### Requirement: Segment selection SHALL apply boundary and stability evidence

系统 SHALL 对每个 segment 首尾各避开 100ms；不足 200ms 的 segment SHALL 将内部窗口收敛到 segment 中点。系统 SHALL 在内部窗口争取至少 3 个有效候选，并优先选择文字与几何位置稳定的文字框轨迹。单字结果 SHALL 在相隔至少 100ms 的两帧出现时视为已确认；若只有一个可见有效帧，系统 SHALL 保留单帧回退；若裁剪窗口没有结果，系统 SHALL 回退到该 segment 未裁剪窗口中的有效候选。任何选择规则 SHALL NOT 读取 transcript 的 ASR 文本或人工参考文本。

#### Scenario: Stable subtitle wins over changing overlay text
- **WHEN** 同一 segment 内字幕文字框在多帧保持相同文本与相近位置，而同带其他文字逐帧变化
- **THEN** 系统选择稳定字幕轨迹中的候选，并将选择证据标记为已确认框轨迹

#### Scenario: One-character subtitle has temporal confirmation
- **WHEN** 相同单字在同一 segment 内至少两帧出现且两帧相隔不小于 100ms
- **THEN** 系统将该单字作为已确认结果，并返回实际所选帧的 PTS

#### Scenario: Short visible subtitle appears only once
- **WHEN** segment 内只有一个通过过滤的非空候选且没有可形成稳定轨迹的第二帧
- **THEN** 系统保留该候选并将选择证据标记为单帧回退

#### Scenario: Trimmed window is empty
- **WHEN** 裁剪后的内部窗口没有非空候选，但同一 segment 未裁剪窗口存在通过过滤的非空候选
- **THEN** 系统选择未裁剪窗口候选并将选择阶段标记为未裁剪回退

### Requirement: Video output SHALL remain compatible and expose selection evidence

系统 SHALL 为每个取得非空选择结果的 segment 返回一条 detection；每条 detection SHALL 保留唯一 `id`、真实 `frame_pts_ms`、归一化 `bbox`、`ocr_text`、`confidence`、`engine_version`、`kind: "subtitle"` 与 `asr_text_prompted: false`，并 SHALL 增加 segment 序号以及包含选择阶段和证据类型的元数据。没有有效结果的 segment SHALL 不生成空 detection。顶层 `media_id` 与 `timebase: "video_ms"` SHALL 保持不变。

#### Scenario: Existing consumer reads upgraded output
- **WHEN** 客户端提交有效视频和 transcript
- **THEN** 返回的顶层字段和既有 detection 字段保持可用，客户端可忽略新增字段继续消费结果

#### Scenario: ASR text is present in transcript
- **WHEN** segment 同时包含时间字段和 ASR 文本
- **THEN** 系统只使用时间字段生成候选，`asr_text_prompted` 为 false，且选择结果不因 ASR 文本内容变化而变化

#### Scenario: Segment has no valid subtitle
- **WHEN** 某 segment 的所有候选均为空或被字幕框过滤规则排除
- **THEN** 响应中不包含该 segment 的空 detection，其他 segment 的结果仍正常返回

### Requirement: Verification SHALL separate deterministic tests from local corpus evaluation

系统 SHALL 提供不依赖私有视频的自动化测试来验证 Provider 选择、显式 CUDA 失败、CPU 回退、候选生成、PTS 保护、区域坐标换算、框过滤、轨迹选择、回退和 API 兼容；系统 SHALL 另提供使用同一组输入分别运行 CUDA 与 CPU 的本地回归评估命令，报告实际 Provider、OCR 吞吐量、预热后 P50/P95 延迟、端到端耗时、1 字、1–2 字、3–4 字召回、完全匹配、CER、空结果和背景误报。评估 SHALL 只使用人工校正 SRT 作为参考文本，并明确 ASR 文本没有被当作 OCR 真值。

#### Scenario: Repository test suite runs without private media
- **WHEN** 开发者在没有 `data/` 私有样本的环境运行测试套件
- **THEN** 所有确定性单元和 API 测试可执行，且不会因缺少私有视频、SRT 或 transcript 而失败

#### Scenario: Frozen corpus is available locally
- **WHEN** 开发者使用本地冻结样本执行 Win11 回归命令
- **THEN** 系统在同一批样本上分别报告 CUDA 与 CPU 的质量和性能，将当前默认配置与已记录基线进行比较，并且不自动回写字幕或生产配置

#### Scenario: GPU recommendation is evaluated
- **WHEN** 同一批冻结样本的 CUDA 与 CPU 对照完成
- **THEN** 结果 SHALL 明确报告 GPU 是否降低预热后延迟或提高吞吐量，以及召回、完全匹配、CER 和背景误报是否保持基线；若 GPU 未实际提速或质量回退，报告 SHALL NOT 将 GPU 声明为推荐路径

### Requirement: OCR SHALL consume ASR output files directly

系统 SHALL 提供可复用的 Python 文件适配层和 Win11 命令行入口，读取 ASR `transcript.json` 的 `schema_version`、`media_id`、`source_file` 与 `segments`，直接使用对应源视频运行现有 OCR 管线。系统 SHALL 在模型初始化和抽帧前校验源视频 SHA-256 与 `media_id` 一致，并 SHALL 以 UTF-8 JSON 原子写出包含相同 `media_id`、`timebase: "video_ms"` 和 `detections` 的证据文件。文件适配层 SHALL NOT 把 ASR 文本传给 OCR 选择器。

#### Scenario: ASR transcript points to its source video
- **WHEN** 调用方传入有效 ASR transcript，且 `source_file` 存在并与 `media_id` 匹配
- **THEN** 系统复用 transcript 的时间段运行视频 OCR，并生成可由 ASR `ocr-propose` 直接读取的 `ocr_evidence.json`

#### Scenario: Source video moved after transcription
- **WHEN** `source_file` 已失效但调用方提供了新的源视频路径，且新文件 SHA-256 与 `media_id` 匹配
- **THEN** 系统使用覆盖路径运行 OCR，不要求修改 transcript

#### Scenario: Transcript and video belong to different media
- **WHEN** 源视频或覆盖视频的 SHA-256 与 transcript 的 `media_id` 不一致
- **THEN** 系统在模型初始化与抽帧前失败，不产生或覆盖 OCR 证据文件

### Requirement: Candidate frames SHALL be decoded by one FFmpeg process

系统 SHALL 在取得候选点后以视频真实帧时间戳将每个请求映射到该时刻之后的第一帧，按实际 PTS 去重，并 SHALL 由单个 FFmpeg 进程输出该请求的全部目标帧。系统 SHALL 逐帧消费批量输出以限制内存占用，并 SHALL 保持 `requested_ms`、真实 `frame_pts_ms` 与 40ms PTS 保护语义。

#### Scenario: Multiple candidate points share an actual frame
- **WHEN** 多个相邻请求点映射到同一真实 PTS
- **THEN** FFmpeg 只输出该帧一次，系统保留距真实 PTS 最近的请求时刻且 OCR 只推理一次

#### Scenario: A video has hundreds of candidate points
- **WHEN** 一次视频 OCR 请求包含多个全片网格和密集窗口候选
- **THEN** 系统只启动一个 FFmpeg 解码进程，并逐张读取临时输出，不把全部未压缩图片同时保存在内存中
