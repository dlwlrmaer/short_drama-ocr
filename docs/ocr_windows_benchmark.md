# Windows 11 短剧 OCR 首轮实测（2026-09-18）

五集、120 张短字幕图的最终选型见[短字幕模型选型](short_subtitle_model_selection.md)。

## 样本与方法

- 机器：Windows 11，Python 3.11.9，RTX 4070 12 GB。OCR 推理均使用 CPU；Qwen3-VL 使用 Ollama 的 CUDA 后端。
- 样本：本机短剧《武当球王》的第 1、2 集，各从 SRT 等间距选 16 条，在时间段中点抽取一帧。视频为 1440×2560、25 fps。视频和 SRT 不包含在仓库中。
- 指标：去掉空格和标点、统一英文字母大小写后，与 SRT 比较完全一致帧数和字符错误率（CER）。SRT 仅作代理参考，尚未逐帧人工核实。
- 延迟为同一进程中第 2 帧起的单帧中位数；不包含 FFmpeg 抽帧、模型下载和加载。

| 方案 | 第 1 集完全一致 / CER | 第 2 集完全一致 / CER | OCR 中位延迟 |
| --- | ---: | ---: | ---: |
| RapidOCR PP-OCRv6 small，整帧 | 15/16 / 0.026 | 未测 | 9.06 秒/帧 |
| RapidOCR PP-OCRv6 small，固定裁剪 | 15/16 / 0.039 | 14/16 / 0.265 | 0.92、0.97 秒/帧 |
| Qwen3-VL 4B 定位字幕带 + RapidOCR | 14/16 / 0.052 | 15/16 / 0.012 | 1.27、1.23 秒/帧 |
| Qwen3-VL 4B 定位后收紧边距 + RapidOCR | 13/16 / 0.078 | 未测 | 1.41 秒/帧 |

PaddleOCR 3.7.0 + PaddlePaddle 3.3.1 的 PP-OCRv5 server 模型在 Windows CPU 默认 oneDNN 路径报 `ConvertPirAttribute2RuntimeAttribute`。设置 `enable_mkldnn=False` 后能运行；已完成的 8 帧有 6 帧完全一致，单帧约 12–14 秒。后续帧耗时异常增长，本轮未得到完整 16 帧结果，不能据此给出整体 CER。当前服务使用的 Tesseract 在此 Windows 环境未安装，所以这轮没有测它。

## 初步结论

本机批量字幕识别优先试 RapidOCR。全帧与裁剪在第 1 集同为 15/16，但裁剪把单帧中位时间从 9.06 秒降到 0.92 秒。短剧字幕位置随片源变化，Qwen3-VL 在每集三帧上定位一次字幕带，再批量复用；第 2 集它排除了画面中的其他文字，效果优于固定区域。第 1 集的自动区域则稍逊，需要先抽样校验，再决定是否用于整集。

本轮只覆盖同一部剧的两集、共 32 个字幕时点。模型选择与区域参数仍需用更多剧集、字幕样式和人工确认的帧真值验证。

## 复现

在仓库根目录用 PowerShell 运行；`ffmpeg`、Ollama 和 `qwen3-vl:4b-instruct` 需可用。首次运行模型需要下载。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install rapidocr==3.9.2 onnxruntime==1.30.0
ollama pull qwen3-vl:4b-instruct

$video = 'E:\samples\drama\1.mp4'
$srt = 'E:\samples\drama\1.srt'
$out = 'data\ocr_benchmark\episode1_vlm'
.\.venv\Scripts\python.exe scripts\benchmark_ocr_win.py locate --video $video --srt $srt --out data\ocr_benchmark\episode1_vlm_roi.json --samples 3
.\.venv\Scripts\python.exe scripts\benchmark_ocr_win.py prepare --video $video --srt $srt --out $out --roi-file data\ocr_benchmark\episode1_vlm_roi.json --limit 16
.\.venv\Scripts\python.exe scripts\benchmark_ocr_win.py run --frames $out --engine rapidocr
```

ROI 文件、截图和逐帧 JSON 保存在 `data/ocr_benchmark/`，该目录被 Git 忽略。若视觉模型定位失败，`locate` 会退出并生成 `.debug.json`；不能把无效坐标传入批量识别。固定区域对照可省去 `locate`，直接运行 `prepare`。

模型及接口参考：[RapidOCR 安装文档](https://github.com/RapidAI/RapidOCRDocs/blob/main/docs/install_usage/rapidocr/install.md)、[PaddleOCR 用法](https://www.paddleocr.ai/v3.3.0/en/version3.x/pipeline_usage/OCR.html)、[Ollama 视觉输入](https://docs.ollama.com/capabilities/vision)、[Qwen3-VL 4B](https://ollama.com/library/qwen3-vl)。Paddle CPU 的 oneDNN 错误见 [PaddleOCR 问题记录](https://github.com/PaddlePaddle/PaddleOCR/issues/18119)。
