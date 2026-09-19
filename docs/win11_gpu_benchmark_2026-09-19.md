# Win11 RTX 4070 OCR Provider 对照（2026-09-19）

## 环境

- Windows 11，Python 3.11.9
- NVIDIA GeForce RTX 4070 12GB，驱动 616.92，CUDA UMD 13.4
- RapidOCR 3.9.2，PP-OCRv6 small
- ONNX Runtime GPU 1.30.0，CUDA/cuDNN 通过 Python GPU extra 预加载
- 样本：`data/ocr_benchmark/short_mix120` 的前 24 张相同字幕裁剪图

`cuda` 模式已确认检测、方向分类、识别三个实际模型会话均以 `CUDAExecutionProvider` 为第一 Provider；`cpu` 模式三个会话均只有 `CPUExecutionProvider`。

## 结果

| 模式 | 实际 Provider | 完全匹配 | 代理 CER | 吞吐量 | 预热后 P50 | 预热后 P95 | 总耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA | CUDAExecutionProvider | 18/24 | 0.4559 | 6.253 帧/秒 | 107.780ms | 262.463ms | 3.838 秒 |
| CPU | CPUExecutionProvider | 18/24 | 0.4559 | 0.188 帧/秒 | 5466.765ms | 5537.647ms | 127.811 秒 |

本轮同环境对照中，CUDA 的完全匹配和代理 CER 与 CPU 一致，吞吐量约为 CPU 的 33.3 倍，P50 延迟约为 CPU 的 1/50.7。因此本机默认推荐 `auto`/CUDA，CPU 保留为兼容回退。

CPU 数据来自安装 `onnxruntime-gpu` 后显式选择 CPU Provider 的同一环境，不能直接替代此前 CPU 专用 `onnxruntime` 包的结果；此前 120 帧报告中 v6 small 的 CPU 中位耗时约 1.071 秒/帧。即便按该旧值比较，本次 CUDA P50 仍约快 9.9 倍。完整 120 帧 CUDA/CPU 对照因 GPU 包内 CPU 路径耗时过长没有继续，当前结论限定为这 24 张冻结样本。

## 真实片源 CUDA 回归

本机外部素材目录 `E:\兼职\瑶瑶如她58` 已确认包含完整的 58 集真实 MP4，以及编号 `01.srt` 到 `58.srt` 的中文字幕；另有一个时间轴和单字文本略有差异的 `5.srt` 备选版本。目录内的时间轴检查报告确认没有字幕结束时间超出视频。此前“没有真实数据”的说法不准确。

已用相邻 ASR 项目的 FunASR 1.4.15 与 Qwen 强制对齐补齐第 31、35、39、50 集 transcript，连同已有第 1、16、48 集组成完整的 7 集输入。2026-09-19 的实际 CUDA 回归确认检测、方向分类、识别三个会话均以 `CUDAExecutionProvider` 为第一 Provider，无回退。

| 配置 | 1–2 字完全匹配 | 3–4 字完全匹配 | 全部完全匹配 | CER | 空结果 | 背景误报 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 秒网格、ROI 0.78、原始框 | 27/41 | 44/57 | 71/98 | 0.4134 | 12 | 12/70 |
| ASR 窗口 250ms 密采样、ROI 0.82、纵向框过滤 | 40/41 | 57/57 | 97/98 | 0.0039 | 1 | 2/70 |
| 再加 100ms 边界裁剪、稳定框轨迹、单字确认 | 41/41 | 57/57 | 98/98 | 0 | 0 | 2/70 |

基础阶段共评估 698 个唯一时间点和 2,094 次三种 ROI OCR，模型推理累计 366.815 秒；推荐 ROI 0.82 的部分为 118.332 秒。边界阶段新增 89 帧，OCR 累计 10.081 秒，包含引擎初始化与解码的墙钟时间为 28.47 秒。最终配置恢复 6/6 条历史漏检，并通过冻结门槛。

本次没有再次执行完整 2,094 次 CPU 推理；同环境 GPU/CPU 质量与速度的直接等价证据仍限定为前述 24 张样本。98/98 CUDA 结果与 2026-09-18 的 CPU 最终结果一致，但两次使用的 ASR transcript 文件并非同一批哈希，因此不将其表述为严格的 Provider 等价实验。

## ASR 文件直连与批量解码

使用第 31 集相同 ASR transcript、相同 PP-OCRv6 small CUDA Provider 和 306 个候选点，对生产 `ocr_from_asr.py` 做了端到端前后对照：

| 解码方式 | FFmpeg 解码进程数 | 端到端耗时 | detections | 与旧 JSON 比较 |
| --- | ---: | ---: | ---: | --- |
| 每个候选单独启动 | 306 | 约 572 秒 | 13 | 基线 |
| 单进程 `select` 批量导出 | 1 | 80.349 秒 | 13 | 完全一致 |

批量方案约加速 7.1 倍。新旧结果的完整 JSON 相等，包含相同文本、真实 PTS、置信度、框和选择证据；因此这次优化只减少重复启动和解析视频的开销，没有改变 OCR 或 segment 选择结果。

原始报告位于 Git 忽略目录：

- `data/ocr_benchmark/provider_gpu24.json`
- `data/ocr_benchmark/provider_cpu24.json`
- `data/yaoyao_single_char_recall_eval/eval_results.json`
- `data/yaoyao_boundary_optimization_eval/eval_results.json`

## 复现

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_win11.ps1 -Mode cuda

.\.venv\Scripts\python.exe scripts\benchmark_provider_win.py `
  --frames data\ocr_benchmark\short_mix120 `
  --modes cuda --limit 24 `
  --out data\ocr_benchmark\provider_gpu24.json

.\.venv\Scripts\python.exe scripts\benchmark_provider_win.py `
  --frames data\ocr_benchmark\short_mix120 `
  --modes cpu --limit 24 `
  --out data\ocr_benchmark\provider_cpu24.json
```
