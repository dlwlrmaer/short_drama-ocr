# NAS OCR

一个可通过 Docker 部署的中英文 OCR API。推送到 `main` 后，NAS 上的 GitHub self-hosted runner 会自动构建并启动服务。

## 本地运行

```bash
docker compose up -d --build
curl http://localhost:8080/health
curl -X POST -F 'file=@test.png' http://localhost:8080/ocr
```

## 视频 ASR 抽帧 OCR

服务需要 FFmpeg，并提供 `/ocr/video` 接口。上传字段为 `video`（视频文件）和
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

## GitHub 部署

将本目录推送到仓库 `dlwlrmaer/short_drama-ocr`，合并到 `main` 即会触发 `.github/workflows/deploy.yml`。
