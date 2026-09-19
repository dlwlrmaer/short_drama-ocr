# Design

## Context

变更动机参见 [proposal.md](proposal.md)。当前 `app/main.py` 把上传、FFmpeg 调用、Tesseract 推理和响应组装放在同一模块；`/ocr/video` 对 ASR segment 扩展 500ms 后按 1000ms 抽帧，并把每帧的全部 OCR 文字合并为 detection。最新评估逻辑存在于 `scripts/yaoyao_*_eval.py`，已经验证候选密度、ROI、框中心过滤、边界裁剪和稳定轨迹，但这些脚本依赖本地目录与缓存结构，不能直接成为服务实现。

当前自动化测试共 27 项且全部通过，主要覆盖输入校验、旧抽帧点、评估指标与选择函数；尚未覆盖 RapidOCR 服务生命周期、ONNX Runtime Provider 选择、ROI 坐标回算、真实 FFmpeg PTS 或 segment 级 API 输出。Win11 主机有 NVIDIA GeForce RTX 4070 12GB，驱动 616.92 报告 CUDA UMD 13.4；当前虚拟环境安装 `onnxruntime==1.30.0`，只暴露 `AzureExecutionProvider` 与 `CPUExecutionProvider`，因此现状仍是 CPU 推理。旧的 `short_subtitle_model_selection.md` 推荐 PP-OCRv5 server，而更新的 7 集/98 条评估在组合完整选择策略后推荐 PP-OCRv6 small；本变更以时间更晚、与目标流水线更接近的结果为默认依据。

## Goals / Non-Goals

**Goals:**

- 让核心候选、识别和选择逻辑成为无私有数据依赖的生产模块，并由 FastAPI 与评估工具共同调用。
- 让 Win11 主机能用固定的 CPU/GPU 互斥依赖和 PowerShell 命令完成安装、CUDA 预热、启动与诊断，优先发挥 RTX 4070 性能。
- 让同一代码库在 Windows 与 Linux 之间按主机自动选择模型和 Provider，并为 2GB 显存 Linux 主机提供不占 GPU 的安全默认值。
- 保持既有 HTTP 基础字段兼容，并让下游能判断每条结果来自稳定轨迹还是回退。
- 以纯函数和可替换的解码/OCR 边界支持快速单元测试，再用独立冒烟测试验证 FFmpeg、真实引擎与实际 CUDA Provider。

**Non-Goals:**

- 使用真实 transcript 量化“全片 1 秒网格”与“保留该网格并在 ASR 窗口内增加 250ms 采样”的候选覆盖差异；ASR 文本不参与 OCR 选择或评分。
- 不用 ASR 文本纠正、提示或评分 OCR，也不自动改写 SRT。
- 不承诺 NAS/Linux 的性能指标，不在 2GB 显存 Linux 主机默认启用 CUDA，不引入 TensorRT、DirectML、Ollama 或动态字幕区域定位。
- 不把《瑶瑶如她》的私有视频、SRT、缓存或逐帧结果提交到仓库。

## Decisions

### 1. 将服务拆为配置、引擎、视频管线和 HTTP 适配层

- `app/settings.py` 保存经校验的默认参数与环境覆盖入口。
- `app/ocr_engine.py` 负责 RapidOCR 生命周期、图片输入和统一的文字框结构。
- `app/video_pipeline.py` 负责候选生成、ROI 坐标换算、框过滤、轨迹聚合、segment 选择与 detection 组装；除解码器与 OCR 适配器外保持纯 Python。
- `app/main.py` 仅保留上传校验、临时文件、错误映射和 HTTP 响应。

这样可以避免把评估脚本整体导入生产服务，也能让评估脚本逐步改用相同模块，防止两份算法漂移。备选方案是在 `main.py` 内直接复制函数，改动较少，但难以隔离模型依赖和构造确定性测试。

### 2. 使用 CUDA 优先的进程级 PP-OCRv6 small provider，并显式验证实际会话

默认引擎使用 `rapidocr==3.9.2` 与经本机验证的 ONNX Runtime GPU 版本。配置提供 `auto`、`cuda`、`cpu`：`auto` 在 `CUDAExecutionProvider` 可用且 PP-OCRv6 small 检测、分类、识别会话均能以 CUDA 为首选 Provider 时使用 GPU，否则记录原因并重建 CPU 引擎；`cuda` 使用同样检查但失败即不 ready；`cpu` 明确关闭 RapidOCR 的 `EngineConfig.onnxruntime.use_cuda`。CUDA 配置使用设备 0，并保留 CPU Provider 作为 ONNX Runtime 不支持算子的会话级后备。

引擎由加锁的进程级 provider 初始化一次，图片与视频共用；启动时执行内存测试图预热。状态不能只检查 `nvidia-smi` 或 `onnxruntime.get_available_providers()`，还要读取 RapidOCR 各 ONNX 推理会话的 `session.get_providers()`，确认 CUDA 是第一 Provider，并记录请求模式、实际 Provider、GPU 名称、模型、包版本和回退原因。`/health` 同时返回 OCR 与 FFmpeg 状态；显式 `cuda` 失败时视频能力不可用。

依赖分为公共、Win11 CPU 和 Win11 CUDA 三组。CPU 路径安装 `onnxruntime`，CUDA 路径安装 `onnxruntime-gpu`；安装脚本先卸载所有 ONNX Runtime 变体，避免共享 `onnxruntime` 模块的包互相覆盖。CUDA 包版本必须与驱动、CUDA 和 cuDNN 主版本兼容，并通过实际会话测试后锁定；可使用 GPU 包提供的 CUDA/cuDNN extras 与 DLL 预加载能力降低手工配置成本。首次模型下载通过单独预热命令完成，正式服务启动不承担不可见的长时间下载。

备选方案包括 DirectML 和 TensorRT。DirectML 当前处于维护模式且本机是 NVIDIA 显卡；TensorRT 会增加引擎构建、缓存与版本矩阵，本轮先采用 RapidOCR 原生支持的 CUDA Execution Provider。每请求创建引擎也被排除，因为会放大模型加载延迟和显存抖动。

运行时在 Provider 之外增加主机 profile。`auto` 按操作系统解析：Windows 选择 `win11`，沿用 PP-OCRv6 small 和 CUDA 优先策略；Linux 选择 `linux-low-vram`，使用 PP-OCRv5 mobile 和 CPU Provider，避免 2GB 显存被 CUDA Runtime 与模型会话占用。模型与执行模式仍允许显式环境变量覆盖，健康检查同时返回请求 profile 和实际 profile。Docker Compose 固定 `linux-low-vram`，Win11 PowerShell 脚本固定 `win11`，避免容器或远程 shell 的平台判断带来歧义。

### 3. 统一候选计划并对每个实际 PTS 只推理一次

先用 `ffprobe` 取得视频时长，再生成 1000ms 全片网格和每个 segment 扩展窗口内的 250ms 点，裁剪到 `[0, duration_ms]` 后合并。每个请求点由 FFmpeg 返回实际 PTS；同一实际帧只做一次 ROI OCR，随后关联到所有覆盖它的 segment。请求点与实际 PTS 的差值保留在候选元数据中，超过 40ms 时不能充当边界补点证据。

全片网格按最新冻结方案保留，即使某些点最终没有关联到 segment。生产管线先用 ffprobe 建立真实帧 PTS 索引，再通过一个 FFmpeg `select` 进程批量导出全部去重目标帧；命令始终使用参数数组，不拼接客户端输入。备选方案是 OpenCV 随机 seek，当前评估虽使用它读取缓存，但 FFmpeg 的实际 PTS 更符合已有 API 时间基准。

### 4. 在 ROI 内推理，随后转换为全画面坐标再过滤

解码帧按 `[0.10, 0.44, 0.85, 0.82]` 裁剪后送入 OCR。引擎输出的多边形或矩形先换算成全画面归一化坐标，再按纵向中心 `0.715–0.815` 过滤。保留下来的框按阅读顺序拼接文字，合并框取所有保留框的外接矩形，置信度使用保留框分数的可解释聚合值。

坐标转换集中在一个函数中并对边界做钳制，避免把 ROI 局部坐标误当作 API 全画面坐标。备选方案是先拼接全部 ROI 文本再做字符串清洗，无法可靠排除同一裁剪区域内的店招和标识。

### 5. 把边界优化实现为确定性的 segment 选择器

每个 segment 建立首尾各裁剪 100ms 的内部窗口；短于 200ms 时收敛到中点。若内部候选少于 3 个，补充起点、中点、终点附近锚点，并受 40ms PTS 保护约束。选择器先按完全一致文字与近似几何位置建立轨迹，再综合出现次数、跨度、置信度、清晰度和距 segment 中点距离排序。

单字轨迹要求至少两帧且间隔 100ms；满足时证据为 `confirmed_box_track`。无法确认但只有一帧清晰可见时返回 `single_frame_fallback`。裁剪窗口为空时从未裁剪 segment 窗口选择并记录 `selection_stage=untrimmed_fallback`。多字稳定轨迹不强制单字确认规则。选择器输入不包含 transcript 文本字段，从类型和调用边界上阻止 ASR 文本参与选择。

备选方案是直接选距中点最近帧，虽然对应初始方案 E，却无法得到边界优化评估中消除空结果和瞬态噪声的收益。

### 6. 兼容旧 detection，并用附加元数据表达 segment 结果

每个有非空结果的 segment 输出一条 detection，沿用原字段；`id` 按 segment 顺序稳定生成，`frame_pts_ms` 与 `bbox` 来自所选候选。新增：

- `segment_index`：输入 `segments` 的零基序号；
- `selection`：包含 `stage`、`evidence`、`candidate_count`、`evidence_span_ms`；
- `requested_ms`：所选候选的请求时间，便于诊断 PTS 偏差。

顶层仍只要求 `media_id`、`timebase` 与 `detections`，不删除旧字段。选择器完全忽略可选 ASR 文本，始终输出 `asr_text_prompted=false`。备选方案是新增 v2 endpoint；当前字段是向后兼容扩展，没有足够理由让客户端迁移 URL。

### 7. 分三层验证，私有冻结样本不进入普通 CI

1. 默认 pytest：用伪 Provider、伪解码器和伪 OCR 覆盖 `auto/cuda/cpu` 选择、禁止静默回退、纯逻辑与 API 错误，不需要 FFmpeg、模型或私有数据。
2. `integration` 标记：临时生成极短视频，验证 ffprobe、FFmpeg 解码、实际 PTS 与坐标管线；CUDA 子集验证模型会话确实以 CUDA 为第一 Provider。缺少相应外部组件时明确跳过。
3. Win11 本地回归命令：在完全相同的冻结缓存/样本和配置下分别运行 CUDA 与 CPU，复用统一生产选择器，输出实际 Provider、OCR 吞吐量、预热后 P50/P95 延迟、端到端耗时、长度分组、CER 与背景误报，并与 2026-09-18 基线比较。

本地回归以人工校正 SRT 为唯一文本参考；真实 ASR transcript 只决定时间窗。GPU 推荐条件是实际 CUDA 会话相对 CPU 降低预热后延迟或提高吞吐量，同时召回、完全匹配、CER 和背景误报不退化；不预设未经实测的加速倍数。备选方案是把私有媒体放进 CI，不符合仓库体积和隐私约束。

### 8. 通过文件适配层直接衔接本机 ASR 输出

`app/asr_adapter.py` 读取 ASR 生成的 UTF-8 `transcript.json`，从 `source_file` 定位源视频，并在执行任何 OCR 前以流式 SHA-256 校验视频与 `media_id` 一致。相对 `source_file` 按 transcript 所在目录解析；文件迁移后允许调用方显式覆盖视频路径，但覆盖文件仍必须通过同一哈希校验。适配层复用统一 segment 校验、RapidOCR Provider 和视频管线，只把 `start_ms`、`end_ms` 传给识别层。

`scripts/ocr_from_asr.py` 作为跨仓库边界，默认把结果原子写到 transcript 同目录的 `ocr_evidence.json`，顶层包含 `schema_version: "1.0"`、相同 `media_id`、`timebase: "video_ms"` 与 `detections`，可直接交给 ASR `ocr-propose`。现有 `/ocr/video` 保持不变，供非本机或不能共享文件系统的客户端使用。备选方案是增加可读取任意本地路径的 HTTP 接口，但这会扩大服务的文件系统访问面，因此本轮采用 Python API 与命令行入口。

### 9. 单进程批量解码全部候选帧

视频管线先用 ffprobe 一次列出视频帧号与 `best_effort_timestamp_time`，按既有“请求时刻之后第一帧”规则将候选点映射到真实帧并按 PTS 去重。随后构造 FFmpeg `select` 表达式，由一个 FFmpeg 进程把所有目标帧输出到请求级临时目录。OCR 逐张读取和释放临时图片，避免同时持有整集未压缩帧；请求结束后目录自动删除。保留单帧 `decode_frame` 供兼容和诊断使用，生产 `process_video` 只调用批量入口。

该方案保留真实 PTS、40ms 保护和每个 PTS 只推理一次的语义，同时消除每个候选点重复启动 FFmpeg 和重复解析视频的开销。若批量进程失败或输出帧数与映射不一致，整次请求返回可诊断错误，避免错位图片被绑定到错误时间戳。

## Risks / Trade-offs

- [全片 1000ms 网格加 250ms 密集窗口仍可能让 GPU 长时间占用] → 对请求点和真实 PTS 双重去重、使用单个 FFmpeg 进程批量解码、每帧只推理一次，并记录候选数和端到端耗时。
- [冻结 ROI 与纵向过滤带只在当前竖屏片源验证] → 参数集中配置并写入引擎状态；本轮保持冻结默认，不宣称适用于其他版式。
- [单帧回退提高召回也可能保留瞬态噪声] → 返回明确证据类型，让下游或人工审核区别处理，并以 70 个共同背景探针守住误报基线。
- [只有 NVIDIA 驱动不代表 CUDA/cuDNN DLL 与 ONNX Runtime GPU 版本兼容] → 以实际模型会话和预热推理作为就绪条件，固定验证过的版本矩阵并输出 DLL/Provider 诊断。
- [GPU 对小 ROI 或单帧请求可能受数据传输开销影响而没有收益] → 用同一批样本比较吞吐量、P50/P95 和端到端耗时，不预设倍数；无实际收益时文档保留 CPU 为推荐回退。
- [GPU 与 CPU 浮点差异可能改变临界置信度结果] → 对同样本比较最终召回、完全匹配、CER 与背景误报，质量退化时不推荐 GPU 默认路径。
- [旧 Docker/NAS 部署会随通用依赖和默认引擎变化] → 保持 HTTP 契约并保留 Docker 构建测试；本轮只将 Win11 验证作为完成门槛，NAS 性能另行验证。
- [ASR 密集窗口会增加候选量和 GPU 占用] → 保留 1 秒全片兜底，记录完整计划、实际评估候选和 OCR 耗时；只将同 ROI、同选择规则下的准确率变化归因于时间表，不声称它降低计算量。
- [ASR transcript 可能指向已移动或错误的视频] → 支持显式路径覆盖，但始终校验源视频 SHA-256 与 `media_id`，不一致时在模型初始化和抽帧前终止。

## Migration Plan

1. 先引入新模块和确定性测试，让旧 endpoint 通过适配器调用，保持响应基础字段。
2. 分离并固定 Win11 CPU/CUDA 依赖，添加 `auto/cuda/cpu` 安装、诊断、预热和启动脚本；在 RTX 4070 上确认 RapidOCR 实际会话使用 CUDA。
3. 接入视频候选、ROI、过滤和选择器，运行默认测试及 FFmpeg 冒烟测试。
4. 用同一冻结样本分别运行 CUDA 和 CPU 回归，核对 98 条短字幕、70 个背景探针及速度指标；将结果记录到评估摘要但不提交媒体。
5. 更新 README 与旧模型选择文档，说明 v6 small 的 Win11 推荐 Provider、CPU 回退和实测边界。

回滚时恢复上一版本应用与依赖文件；HTTP URL 和旧基础字段未迁移，无需转换持久化数据。
