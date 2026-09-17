# NAS OCR

一个可通过 Docker 部署的中英文 OCR API。推送到 `main` 后，NAS 上的 GitHub self-hosted runner 会自动构建并启动服务。

## 本地运行

```bash
docker compose up -d --build
curl http://localhost:8080/health
curl -X POST -F 'file=@test.png' http://localhost:8080/ocr
```

API 文档：`http://NAS-IP:8080/docs`

## GitHub 部署

将本目录推送到仓库 `dlwlrmaer/short_drama-ocr`，合并到 `main` 即会触发 `.github/workflows/deploy.yml`。
