# Tasks

## 1. 运行时与配置基础

- [x] 1.1 将公共依赖、Win11 CPU 的 `onnxruntime` 和 Win11 CUDA 的 `onnxruntime-gpu` 拆成互斥依赖组，实测并锁定与 RTX 4070 驱动/CUDA/cuDNN 兼容的版本，移除应用对系统 Tesseract 的硬依赖，并分别验证全新 Python 3.11 虚拟环境可导入 CPU 与 CUDA Provider
- [x] 1.2 新增集中配置模块，定义并校验 `auto/cuda/cpu` 执行模式、GPU 设备 0、ROI、框中心带、1000ms/250ms 采样、500ms 扩窗、100ms 边界裁剪、40ms PTS 保护与 100ms 单字确认默认值，并用执行模式及参数边界单元测试验证
- [x] 1.3 实现进程级 RapidOCR PP-OCRv6 small provider、内存图预热和实际模型会话 Provider 检查，并用替身会话测试单例复用、`auto` CUDA 优先、`auto` CPU 回退、显式 `cuda` 失败和显式 `cpu` 禁用 CUDA

## 2. 视频候选与字幕框识别

- [x] 2.1 实现 ffprobe 时长读取以及“全片 1000ms 网格 + segment 扩窗内 250ms 采样”的去重候选计划，并用短 segment、重叠窗口、零时刻和视频尾部测试验证
- [x] 2.2 将 FFmpeg 解码封装为返回请求时间、真实 PTS 和图片的适配器，按实际 PTS 去重推理，并用模拟 subprocess 测试超时、非零退出、缺少 PTS 与 40ms 偏差标记
- [x] 2.3 实现冻结 ROI 裁剪、RapidOCR 文字框统一、全画面坐标回算、`0.715–0.815` 中心过滤和阅读顺序拼接，并用多分辨率及越界框测试验证文本、置信度和归一化合并框

## 3. Segment 选择与输出契约

- [x] 3.1 实现首尾各裁剪 100ms、短 segment 中点收敛、至少 3 个内部候选及边界锚点规划，并用 40ms PTS 保护测试验证候选资格
- [x] 3.2 实现按精确文本和几何位置聚合的稳定框轨迹排序，以及单字至少间隔 100ms 的双帧确认，并复用评估用例验证稳定字幕优先于变化标识
- [x] 3.3 实现 `single_frame_fallback` 与 `untrimmed_fallback`，并用单帧可见、裁剪为空、全程为空和多字轨迹测试验证证据与选择阶段
- [x] 3.4 实现每个非空 segment 一条 detection 的组装，保留旧字段并增加 `segment_index`、`requested_ms` 与 `selection`，用契约测试验证唯一 ID、真实 PTS、全画面 bbox 和 `asr_text_prompted=false`

## 4. FastAPI 集成与兼容

- [x] 4.1 将 `/ocr` 切换到共享 RapidOCR provider，保持 `filename`/`text` 与既有 4xx 行为，并运行现有图片接口测试及新增无框结果测试
- [x] 4.2 将 `/ocr/video` 接入统一候选、解码、ROI OCR 与 segment 选择管线，保持上传校验、SHA-256、临时文件清理和顶层响应字段，并用伪视频/伪引擎 API 测试验证多 segment、空 segment 与 ASR 文本变化不影响结果
- [x] 4.3 扩展 `/health` 的请求模式、实际 Provider、GPU 设备、模型、回退原因、OCR/FFmpeg 和视频就绪状态，并用真实 Provider、伪 CUDA 回退、显式 CUDA 失败、FFmpeg 缺失和模型初始化失败测试验证状态码与响应内容

## 5. Win11 开发与运行工具

- [x] 5.1 添加支持 `auto/cuda/cpu` 的 PowerShell 安装脚本，以创建 Python 3.11 虚拟环境、清理冲突的 ONNX Runtime 变体、安装对应固定依赖并检查 NVIDIA/FFmpeg，使用三种模式和重复执行验证依赖选择与幂等性
- [x] 5.2 添加模型预热和本地启动脚本，在 RTX 4070 上验证检测、分类、识别会话均以 CUDA 为第一 Provider，并以 `/health` 和合成图片确认无需 Docker、WSL、Ollama 或 Tesseract 即可完成 GPU OCR；另验证 CPU 模式可独立运行
- [x] 5.3 更新 README 的 Win11 CPU/GPU 安装、启动、Provider 验证、请求、配置和 CUDA DLL 故障排查说明，并修订旧模型选择文档，使其明确 PP-OCRv6 small 是组合流水线默认值、PP-OCRv5 server 仅为旧样本结论，GPU 是否推荐以同样本实测为准

## 6. 分层验证与评估一致性

- [x] 6.1 让现有评估脚本复用生产候选、坐标和选择模块或加入等价性测试，验证脚本冻结用例与生产函数输出一致，避免两套算法漂移
- [x] 6.2 添加独立 `integration` 冒烟测试，临时生成短视频并验证 ffprobe、FFmpeg 解码和真实 PTS；增加 CUDA 标记测试以确认 RapidOCR 三类模型会话的第一 Provider，缺少对应外部组件时明确跳过且默认 pytest 不依赖 GPU 或私有媒体
- [x] 6.3 在当前 Win11 主机运行完整默认测试并确认不少于现有 27 项全部通过，再分别以 `cuda` 和 `cpu` 运行图片与视频 API 冒烟命令，记录 Python、RapidOCR、ONNX Runtime、CUDA/cuDNN、驱动、GPU、FFmpeg、模型与实际 Provider
- [x] 6.4 在同一批本地 OCR 样本上分别运行 CUDA 与 CPU 性能对照，记录 OCR 吞吐量、预热后 P50/P95 延迟和端到端耗时；只有实际 CUDA 会话提速且最终质量不退化时才在文档中将 GPU 标为推荐路径
- [x] 6.5 补齐《瑶瑶如她》7 集真实 transcript，并在 CUDA 实际会话上重建冻结缓存，验证并记录 1 字、1–2 字、3–4 字、全部 98 条、70 个背景探针和耗时；与既有 CPU 最终指标比较时记录 transcript 哈希不同的限制，GPU/CPU 严格等价结论只采用同样本直接对照
- [x] 6.6 运行 `openspec-cn validate add-win11-rapidocr-subtitle-pipeline --strict` 和 `openspec-cn status --change add-win11-rapidocr-subtitle-pipeline`，确认规范校验通过且所有已完成任务状态一致

## 7. ASR 文件直连

- [x] 7.1 提取共享 transcript 校验并实现 ASR 文件适配层，解析 `source_file`、校验 `schema_version` 与 `media_id`，在 OCR 前核对源视频 SHA-256，复用生产视频管线且不传入 ASR 文本
- [x] 7.2 添加 Win11 命令行入口，支持 `auto/cuda/cpu`、源视频路径覆盖和输出路径覆盖，默认原子写出 transcript 同目录的 `ocr_evidence.json`
- [x] 7.3 添加相对路径、哈希不一致、路径覆盖、输出契约及原子写入测试，更新 README 后运行默认测试与 OpenSpec 严格校验；另以第 31 集真实 ASR transcript 在 CUDA Provider 上完成端到端验证并通过 ASR 侧契约校验

## 8. FFmpeg 单进程批量解码

- [x] 8.1 实现帧级 ffprobe 时间戳索引和候选到“请求时刻之后第一帧”的映射，按真实 PTS 去重并保留最近请求时刻
- [x] 8.2 实现单个 FFmpeg `select` 进程批量导出目标帧，由生产视频管线逐张消费临时图片并保持 PTS 保护及 OCR 去重语义
- [x] 8.3 添加单进程调用、PTS 映射、重复帧、错误处理及真实 FFmpeg 集成测试，并在第 31 集相同 transcript 上复跑 CUDA 端到端对照；306 个候选从约 572 秒降至 80.349 秒，13 条 detection 的新旧 JSON 完全一致

## 9. Windows 与 Linux 动态 profile

- [x] 9.1 增加 `auto/win11/linux-low-vram` 运行 profile、模型配置和环境变量覆盖；Windows 默认 PP-OCRv6 small 与 CUDA 优先，Linux 默认 PP-OCRv5 mobile 与 CPU，并在健康状态中公开实际配置
- [x] 9.2 将 Win11 脚本和 Docker Compose 分别固定到对应 profile，扩展 ASR 命令行、README 与单元测试，并真实预热两种模型验证实际 Provider
