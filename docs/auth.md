# 认证说明

> 若需完整配置示例，请查阅根目录 `DEPLOYMENT_GUIDE.md`。

系统提供简单的 API Key 认证方式。在 `.env` 文件中设置 `API_KEY` 后，客户端需要在请求头中使用 `X-API-Key` 传递该值。

