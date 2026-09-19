# Windows 11 短剧短字幕 OCR 选型（2026-09-18）

## 最新状态

本文记录的是较早的 120 帧“单帧模型”对照，其中 PP-OCRv5 server 优先避免漏读。后续《瑶瑶如她》7 集/98 条组合流水线评估加入密集候选、字幕框过滤和边界选择后，Win11 默认已更新为 **PP-OCRv6 small**，详见[最新评估结论](yaoyao_ocr_conclusions_2026-09-18.md)。GPU 或 CPU Provider 不改变模型选择；GPU 是否成为本机推荐路径，以同样本的实际 Provider、速度和最终质量对照为准。

## 当时决定

**当时的主模型选 RapidOCR 的 PP-OCRv5 server（检测与识别均用 server ONNX 模型）。**本次选型优先避免短字幕漏读。字幕位置先按片源确定，再在该区域内识别；区域内有道具文字或店招时，还需要按文字框位置筛掉非字幕。快速预览选 PP-OCRv6 small。

这里的“server”是模型规格，不表示需要远端服务器；本次均在 Windows 11 本机 CPU 上运行。没有把 OCR 当成字幕定位器：我先直接查看完整视频帧，随后核对已有手动框选记录，并用相同裁剪图比较模型。

## 选集与样本

`scripts/rank_short_subtitle_episodes.py` 扫描 `E:\兼职` 中 392 个有同集视频与中文字幕 SRT 的集数，统计每集 1–4 字字幕。选用下列五集，每集从所有 1–4 字字幕中等距抽 24 条，并在 SRT 时段中点取一帧，共 120 帧；其中 38 帧为 1–2 字。跳过排名靠前但参考 SRT 来自 OCR 输出的集数，以降低参考文本与被测 OCR 同源的偏差。

| 片源 | 集数 | 该集 1–4 字条数 | 其中 1–2 字 | 测试帧 |
| --- | ---: | ---: | ---: | ---: |
| 下山即无敌，从保镖开始封神 | 1 | 86 | 38 | 24 |
| 下山即无敌，从保镖开始封神 | 2 | 62 | 23 | 24 |
| 下山即无敌，从保镖开始封神 | 6 | 54 | 17 | 24 |
| 无敌女鬼有点恋爱脑 | 5 | 52 | 6 | 24 |
| 武当球王 | 7 | 43 | 22 | 24 |

《下山即无敌》的区域参考 `ocr/ocr_workspace.json` 中的 `reference_roi={x:52,y:1060,width:922,height:406}`；《无敌女鬼》的参考 `中文/ocr_workspace.json` 中的 `{x:20,y:1082,width:1028,height:386}`。按参考画布 1080×1920 换算到各集视频；《武当球王》用直接查看视频帧确定的区域。原始视频、SRT、裁剪图与逐帧 JSON 在本机 `data/ocr_benchmark/`，不提交到仓库。可分享的逐帧文本和耗时见 [短字幕结果 CSV](short_subtitle_results.csv)。

## 同图对比

四种候选模型都在上述 120 张相同裁剪图上测试。完全一致与 CER 都是去除空格和标点后的 **SRT 代理指标**；“包含参考字幕”允许 OCR 同时读到背景文字，用于区分漏读和杂字。耗时为同一进程第二张起的单张中位数，不含抽帧或模型下载。越高越好的指标是完全一致、包含和 1–2 字命中；CER 与耗时越低越好。

| RapidOCR 模型 | 完全一致 | 包含参考字幕 | 1–2 字完全一致 | 空结果 | SRT 代理 CER | CPU 中位耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PP-OCRv6 small | 102/120 | 109/120 | 31/38 | 3 | 0.252 | 1.071 秒 |
| PP-OCRv5 mobile | 101/120 | 107/120 | **33/38** | 2 | 0.309 | **0.915 秒** |
| **PP-OCRv5 server** | **105/120** | **117/120** | **33/38** | **0** | 0.428 | 4.117 秒 |
| PP-OCRv6 medium | 104/120 | 115/120 | 31/38 | 1 | 0.431 | 4.249 秒 |

| 片源 / 集数 | v6 small | v5 mobile | v5 server | v6 medium |
| --- | ---: | ---: | ---: | ---: |
| 下山即无敌 1 | 17/24 | 17/24 | 19/24 | 19/24 |
| 下山即无敌 2 | 20/24 | 20/24 | 21/24 | 20/24 |
| 下山即无敌 6 | 23/24 | 23/24 | 23/24 | 23/24 |
| 无敌女鬼 5 | 20/24 | 20/24 | 18/24 | 18/24 |
| 武当球王 7 | 22/24 | 21/24 | 24/24 | 24/24 |

PP-OCRv5 mobile 在先前较紧的区域试验中只有 15/40 完全一致；换成已有手动 ROI 后，同样抽样的 40 帧升至 34/40，故补测完整 120 帧。它对裁剪宽度很敏感，不能仅凭紧裁样本淘汰。

## 错误复核与使用边界

- v5 server 对 120 帧没有空结果，真实字幕出现在输出中的次数也最多，适合优先保住一两个字的对白。
- 它的 CER 比 v6 small 高，主要因为裁剪范围内还出现传单、店招、服饰等文字。例如“我接了”一帧同时有传单小字；v5 server 读对了字幕，也读出了整张传单。**直接拼接所有 OCR 行并非最终字幕输出**，应先利用位置框保留字幕行。
- “可惜了”一帧同时显示正在退出的“可期了”和新字幕；“咦”一帧处在模糊转场。它们的 SRT 只记录一个时刻的文本，不能当成干净的逐帧真值。将两帧从逐字指标中去掉，不改变四种模型的排序。
- v5 mobile 每帧约快 4.5 倍，1–2 字逐字命中与 server 相同，但整体包含参考字幕数少 10 帧。v6 small 杂字较少，CER 最低；两者都可作快速预览。v6 medium 耗时与 v5 server 接近，在本样本未取得更好的短字幕命中。

本次结论针对这五集、同一套字幕区域及 Windows CPU。完整片源没有逐帧人工转写真值；SRT 时间偏移、繁简字形和动画帧会影响代理指标。此前原生 PaddleOCR Windows CPU 测试还遇到 oneDNN 问题，详见[首轮报告](ocr_windows_benchmark.md)。

## 复现

```powershell
.\.venv\Scripts\python.exe scripts\rank_short_subtitle_episodes.py --root 'E:\兼职' --out data\ocr_benchmark\short_episode_ranking.json
.\.venv\Scripts\python.exe scripts\benchmark_ocr_win.py prepare --video 'E:\兼职\F下山即无敌，从保镖开始封神\成片\4.27配音版\01.mp4' --srt 'E:\兼职\F下山即无敌，从保镖开始封神\中文\1.srt' --out data\ocr_benchmark\short_down1 --limit 24 --max-chars 4 --x0 0.048148 --y0 0.552083 --x1 0.901852 --y1 0.763542
.\.venv\Scripts\python.exe scripts\benchmark_ocr_win.py merge --sources data\ocr_benchmark\short_down1 data\ocr_benchmark\short_down2 data\ocr_benchmark\short_down6 data\ocr_benchmark\short_ghost5 data\ocr_benchmark\short_wudang7 --out data\ocr_benchmark\short_mix120
.\.venv\Scripts\python.exe scripts\benchmark_ocr_win.py run --frames data\ocr_benchmark\short_mix120 --engine rapidocr_v5_server
```

其余四集的 SRT、视频和归一化区域见本机对应 `data/ocr_benchmark/short_*/manifest.json`；`merge` 前需依照第一条 `prepare` 的方式分别准备。模型规格参考 [RapidOCR 模型列表](https://github.com/RapidAI/RapidOCRDocs/blob/main/docs/model_list.md)。
