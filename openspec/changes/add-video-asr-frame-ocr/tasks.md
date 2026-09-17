## 1. OpenSpec 与契约

- [x] 1.1 创建视频 ASR 抽帧 OCR 的 OpenSpec proposal/spec/design，并通过 `openspec-cn validate --strict` 验证制品结构
- [x] 1.2 固化 transcript 输入校验、抽帧窗口和 OCR 回传字段的测试契约

## 2. 核心实现

- [x] 2.1 实现可测试的 ASR segment 校验、500ms 扩展窗口、1000ms 抽帧点生成与去重排序
- [x] 2.2 实现临时文件上传、SHA-256 media_id、FFmpeg 逐点解码和错误映射
- [x] 2.3 实现视频 OCR endpoint，并保持图片 `/ocr` 的响应兼容

## 3. 验证与交付

- [x] 3.1 添加抽帧、输入错误、FFmpeg 错误和 endpoint 兼容性测试，并运行 pytest
- [x] 3.2 更新 requirements、Dockerfile、README 与 API 示例，验证 Docker 配置包含 FFmpeg
- [x] 3.3 完成 OpenSpec 任务勾选并运行 OpenSpec 状态/严格校验命令
