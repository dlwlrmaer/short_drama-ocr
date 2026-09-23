# NAS OCR

一个面向短剧字幕的中英文 OCR API。服务按操作系统动态选择 OCR profile，也可以用环境变量固定配置；推送到 `main` 后，NAS 上的 GitHub self-hosted runner 仍可自动构建并启动 CPU 服务。

## 动态运行配置

| `OCR_RUNTIME_PROFILE` | 默认模型 | 默认执行模式 | 用途 |
| --- | --- | --- | --- |
| `auto` | 按系统与硬件选择 | 按系统与硬件选择 | 默认值；检测系统内存、NVIDIA 显存和操作系统 |
| `win11` | PP-OCRv6 small | `auto`，CUDA 可用时优先 | Windows 11 开发机和 RTX 显卡 |
| `linux` | PP-OCRv6 small | `auto`，CUDA 可用时优先 | Linux 且 NVIDIA 显存不少于 4GB |
| `linux-low-vram` | PP-OCRv5 mobile | `cpu` | Linux 小主机；默认不占用 2GB 显存 |

配置优先级为显式环境变量高于 profile 默认值。可用变量：

- `OCR_RUNTIME_PROFILE=auto|win11|linux|linux-low-vram`
- `OCR_EXECUTION_MODE=auto|cuda|cpu`，覆盖 profile 的执行模式
- `OCR_MODEL_PROFILE=ppocrv6-small|ppocrv5-mobile`，覆盖 profile 的模型
- `OCR_GPU_DEVICE_ID=0`，指定 CUDA 设备
- `OCR_BACKGROUND_SUPPRESSION=off|spatial|adaptive`，控制字幕背景硬遮罩
- `OCR_MAX_BATCH_IMAGES=1..64`，覆盖根据系统内存计算的批量上限

`auto` 在 Windows 选择 Win11 配置；在 Linux 上检测到不少于 4GB NVIDIA 显存时选择 `linux`，否则选择 `linux-low-vram`。系统内存低于 8GB、8–16GB 和更高配置时，默认单批上限分别为 8、16 和 32 张；低显存 profile 最高默认 16 张。`/health` 会返回 `requested_profile`、`active_profile`、`hardware_tier`、系统/显存、批量上限、背景抑制、模型和实际 Provider。

Linux 2GB 显存机器若明确想尝试 GPU，可设置 `OCR_EXECUTION_MODE=cuda`；CUDA 初始化或显存不足时会直接报错，不会伪装成 GPU 推理。

## Windows 11 本机运行

前置条件：Python 3.11 x64、FFmpeg（`ffmpeg` 和 `ffprobe` 均在 PATH）以及最新 NVIDIA 驱动。RTX 显卡推荐 CUDA 模式；无需 Docker、WSL、Ollama 或 Tesseract。

```powershell
# 自动检测 NVIDIA 显卡并安装 GPU 或 CPU 依赖
powershell -ExecutionPolicy Bypass -File scripts\setup_win11.ps1 -Mode auto

# 显式 GPU 安装与预热；CUDA 不可用时直接报错
powershell -ExecutionPolicy Bypass -File scripts\setup_win11.ps1 -Mode cuda

# 启动服务
powershell -ExecutionPolicy Bypass -File scripts\run_win11.ps1 -Mode auto

# 本机健康、图片和生成视频冒烟测试
$env:OCR_EXECUTION_MODE = 'cuda'
.\.venv\Scripts\python.exe scripts\smoke_win11.py `
  --image data\ocr_benchmark\short_mix120\short_down1_frame_001.png
```

执行模式：

- `auto`：实际 CUDA 模型会话可用时使用 GPU，否则记录原因并回退 CPU。
- `cuda`：要求检测、分类和识别会话均以 `CUDAExecutionProvider` 为第一 Provider，禁止静默回退。
- `cpu`：明确使用 `CPUExecutionProvider`。

访问 `http://localhost:8080/health` 检查 `requested_mode`、`actual_provider`、三类模型的 `session_providers`、GPU 名称和回退原因。看到 NVIDIA 驱动或安装了 `onnxruntime-gpu` 并不等于实际使用 GPU，应以这里的模型会话 Provider 为准。

若 CUDA 初始化失败，先重新运行 `setup_win11.ps1 -Mode cuda` 清理冲突的 ONNX Runtime 包。仍失败时检查健康响应中的 DLL/Provider 错误、NVIDIA 驱动以及 CUDA/cuDNN 版本。需要临时恢复服务可使用 `run_win11.ps1 -Mode cpu`。

## 本地运行

```bash
# Linux 直接运行：自动选择低显存 CPU profile
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-cpu.txt
OCR_RUNTIME_PROFILE=auto uvicorn app.main:app --host 0.0.0.0 --port 8080

# Docker Compose 自动检测容器可见的硬件；无可见 GPU 时使用低显存 CPU profile
docker compose up -d --build
curl http://localhost:8080/health
curl -X POST -F 'file=@test.png' http://localhost:8080/ocr

# 输入是完整视频帧时，启用字幕 ROI、背景遮罩和字幕框过滤
curl -X POST -F 'file=@frame.png' 'http://localhost:8080/ocr?subtitle=true'

# 批量图片，保持输入顺序返回；上限由硬件配置决定
curl -X POST 'http://localhost:8080/ocr/batch?subtitle=true' \
  -F 'files=@frame-001.png' \
  -F 'files=@frame-002.png'
```

单图接口继续返回 `{filename, text}`。批量接口返回 `{count, results}`，其中每条结果包含 `index`、`filename` 和 `text`。批量处理逐张解码和推理，共用同一个已预热模型，不会同时把整批未压缩图片留在内存中。`subtitle=true` 表示输入是完整视频帧：程序会应用字幕 ROI、空间背景遮罩和字幕框过滤；普通文档图片不要开启。

## OCR 与 ASR 双服务调用

OCR 和相邻 `short_drama-asr` 仓库分别有自己的 Dockerfile、Compose 文件和端口，可以独立构建、重启。OCR 保持原有宿主机 `0.0.0.0:8080` 监听，ASR 默认只监听 `127.0.0.1:8090`；需要跨主机调用 ASR 时设置 `ASR_BIND_ADDRESS=0.0.0.0`。OCR 的 `OCR_RUNTIME_PROFILE=auto` 在容器中按可见硬件选型；2 GB 显存且未透传 GPU 时使用 CPU 和轻量模型。

```powershell
docker compose -f E:\short_drama\asr\docker-compose.yml up -d --build
docker compose -f E:\short_drama\ocr\docker-compose.yml up -d --build
curl.exe -f http://localhost:8090/health
curl.exe -f http://localhost:8080/health

# 用同一视频先生成 ASR JSON，再以该 JSON 的时间轴引导视觉 OCR
curl.exe -f -F "file=@E:\media\episode.mp4" http://localhost:8090/asr/transcribe -o transcript.json
curl.exe -f -F "video=@E:\media\episode.mp4" -F "transcript=@transcript.json;type=application/json" http://localhost:8080/ocr/video -o ocr_evidence.json
```

`POST /ocr/video` 默认以 `sequence` 模式扫描语音附近和全片画面，返回每条视觉字幕的出现时间与文字；`?mode=legacy` 可使用旧版按 ASR 段抽帧。ASR 文字只决定采样时间，不作为 OCR 识别提示。接口会对比 transcript 中的 `media_id` 与上传视频的 SHA-256，避免两个服务误配素材。返回的 `ocr_evidence.json` 可交给 ASR 的 `ocr-propose` 继续做人工确认的同音字纠错。单张图片和批量图片仍分别调用 `/ocr`、`/ocr/batch`。

OCR 容器使用 CPU 依赖；若要在 Docker 中使用 NVIDIA GPU，需要另外安装适配的 GPU 运行时与 ONNX Runtime 依赖，并提供 GPU 透传。Win11 主机也可继续使用本机 `setup_win11.ps1` 的 CUDA 方案。

## 视频 ASR 抽帧 OCR

服务需要 FFmpeg。Win11 上可直接把 ASR 生成的 `transcript.json` 传给 OCR；程序会读取其中的 `source_file`，校验它的 SHA-256 与 `media_id` 一致，并在 transcript 同目录原子写出可供 ASR `ocr-propose` 使用的 `ocr_evidence.json`：

```powershell
.\.venv\Scripts\python.exe scripts\ocr_from_asr.py `
  E:\short_drama\asr\output\episode\transcript.json `
  --profile win11 `
  --mode cuda

# source_file 已迁移时可以显式覆盖，但仍会核对 media_id
.\.venv\Scripts\python.exe scripts\ocr_from_asr.py `
  E:\short_drama\asr\output\episode\transcript.json `
  --video E:\media\episode.mp4 `
  --output E:\short_drama\asr\output\episode\ocr_evidence.json `
  --mode auto
```

Python 调用方也可以直接调用 `app.asr_adapter.run_asr_transcript()` 和 `save_evidence()`。适配层只读取 segment 的时间字段，不会把 `asr_text` 作为识别提示。

目录中没有可信字幕文件时，可以直接从视频音轨生成独立 ASR 时间轴，再批量执行 OCR。下面的流程不读取素材目录中的 SRT，ASR 文字也不会传给 OCR：

```powershell
# ASR 环境需要 funasr、modelscope 和 CUDA PyTorch
C:\Users\user\.venvs\short-drama-asr\Scripts\python.exe `
  scripts\transcribe_asr_batch.py E:\media\drama `
  --engine funasr --model paraformer-zh

# OCR 环境自动选择当前 Win/Linux 硬件配置
.\.venv\Scripts\python.exe scripts\ocr_asr_batch.py `
  E:\media\drama\ocr\asr
```

ASR 脚本复用一个模型实例，并按词时间戳生成字幕尺度的候选窗。OCR 脚本默认启用 `sequence` 模式：按 Win11 250ms / Linux 450ms 扫描语音附近画面，同时按 Win11 450ms / Linux 700ms 扫描全片。每次字幕文字变化都可产生独立条目；同一句在画面消失后再次出现也会拆开。同一 ASR 窗中的多条字幕和没有 ASR 窗的短句均可检出。ASR 文字不参与识别，素材目录中的 SRT 也不会被读取。字幕中的 `亖` 和 `三` 按当前短剧要求统一替换成 `死`，原字保存在 evidence 的 `raw_ocr_text` 中。

默认结果放在 `ocr/refined`，保留旧版结果供对照。脚本支持 `--episodes 1 5-10`、`--speech-ms`、`--fallback-ms`、`--mode legacy`、逐集原子落盘和断点续跑；输出包含 `subtitles/*.srt`、`evidence/*.json`、`subtitle_roi.json` 和 `ocr_report.json`。`sequence` 模式中的 `detected_segments` 是视觉字幕条数，不宜再当作 ASR 窗覆盖率。

序列模式会过滤短暂的字母数字花纹误识别。对升级前已生成的序列结果，可运行 `python scripts/filter_sequence_noise.py E:\media\drama\ocr\refined` 重新生成 SRT；被剔除的候选仍保留在 evidence 的 `rejected_noise` 中。此步骤只读取 OCR 证据，不读取素材自带字幕。

素材自带字幕只用于事后评估时，可运行：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_sequence_recall.py `
  E:\media\drama\字幕文件 `
  E:\media\drama\ocr\evidence `
  E:\media\drama\ocr\refined\evidence `
  --output E:\media\drama\ocr\refined\recall_comparison.json
```

生产视频管线会先将候选时刻映射到真实帧 PTS，再由一个 FFmpeg 进程批量导出全部去重候选。第 31 集的 306 个候选在 RTX 4070 上从逐点解码约 572 秒缩短到 80.349 秒，输出的 13 条 detection 与优化前 JSON 完全一致。

对具有 `avg_frame_rate == r_frame_rate` 和 `nb_frames` 的固定帧率视频，管线直接由帧率元数据计算候选帧编号，避免先用 FFprobe 扫描全部帧；其他视频自动回退到逐帧 PTS 映射。

服务另外保留 `/ocr/video` 上传接口。上传字段为 `video`（视频文件）和
`transcript`（参考 `short_drama-asr` 的 UTF-8 JSON，至少包含带 `start_ms`、
`end_ms` 的 `segments`）。每条片段前后扩展 500ms，约每秒抽取一帧，返回与
ASR OCR 契约兼容的 `media_id`、`timebase` 和 `detections`：

```bash
curl -X POST http://localhost:8080/ocr/video \
  -F 'video=@episode.mp4;type=video/mp4' \
  -F 'transcript=@transcript.json;type=application/json'
```

视频不会覆盖上传的原文件，服务会在请求结束后清理临时副本。`/ocr` 图片接口保持不变。

API 文档：`http://NAS-IP:8080/docs`

Windows 11 短剧 OCR 模型选型见 [短字幕测试结论](docs/short_subtitle_model_selection.md)，首轮区域实验见 [测试报告](docs/ocr_windows_benchmark.md)。

RTX 4070 的实际 CUDA/CPU Provider 对照见 [Win11 GPU 测试](docs/win11_gpu_benchmark_2026-09-19.md)。24 张同样本中 CUDA 与 CPU 质量指标一致，CUDA P50 约 108ms，当前本机推荐 `auto`/CUDA。

## Win11 GPU/CPU 对照

在同一份已准备的帧样本上运行：

```powershell
$env:OCR_EXECUTION_MODE = 'cuda'
.\.venv\Scripts\python.exe scripts\benchmark_provider_win.py `
  --frames data\ocr_benchmark\short_mix120 `
  --modes cuda cpu `
  --out data\ocr_benchmark\provider_comparison.json
```

报告包含实际 Provider、吞吐量、预热后 P50/P95、端到端耗时、完全匹配和代理 CER。只有 CUDA 实际提速且质量指标不退化时，才把 GPU 标为当前机器的推荐路径。

## GitHub 部署

将本目录推送到仓库 `dlwlrmaer/short_drama-ocr`，合并到 `main` 即会触发 `.github/workflows/deploy.yml`。
