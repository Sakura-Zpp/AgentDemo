# 运维手册

## 启动前检查

1. 使用 Python 3.11–3.13，执行 `python -m pip install -e ".[dev]"`。
2. 配置 `DASHSCOPE_API_KEY`；不要把真实密钥写入 YAML、JSON 或版本库。
3. 多实例部署需设置 `AGENT_CHECKPOINT_BACKEND=postgres` 和 `AGENT_CHECKPOINT_DSN`。
4. 运行 `python -m pytest -W error -p no:cacheprovider -m "not e2e"`、`python -m ruff check .` 和 `python -m mypy`。

## 健康与指标

- HTTP 健康端点：`/_stcore/health`，`healthcheck.py` 在非 200 或连接失败时返回 1。
- 页面侧边栏“运行状态”显示初始化状态、Checkpoint 后端、工具数、请求/路由/工具计数与延迟。
- 重点告警：`requests.error`、`requests.timeout` 持续增长，或工具错误率、P95 延迟异常。
- 进程内指标会随进程重启清零；生产环境应由外部采集器读取并持久化。

## 常用操作

- 启动：`streamlit run app.py --server.fileWatcherType=none`。关闭生产文件监视可避免第三方动态模块触发无害但嘈杂的扫描异常；开发热重载时可省略该参数。
- 容器：`docker compose up --build`
- 删除当前会话：页面侧边栏“删除当前聊天”。
- 删除当前用户记忆：勾选确认后点击“清除长期记忆”。
- 停用故障 MCP：在本地 MCP 配置中设置 `isActive: false` 后重启进程。

## 数据与容量

- SQLite Checkpoint 默认保留 30 天、最多 500 个 thread。
- 长期记忆默认保留 180 天、每用户最多 200 条。
- 日志按天轮转并脱敏；仍应由平台设置文件保留和磁盘水位告警。
- Chroma、BM25 与摘要清单必须作为同一个索引版本备份和恢复。

## 已知限制

- URL 中的 `user` 是 Demo 隔离标识，不是认证；公开部署必须接入登录身份。
- SQLite 长期记忆只适合单实例；多实例必须使用共享记忆实现。
- 进程内指标不是完整的生产监控后端。
