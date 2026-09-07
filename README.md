# AgentDemo

基于 Streamlit、LangChain 和 LangGraph 的多智能体企业知识库 Demo。

## 架构

用户请求首先进入 LangGraph Supervisor，再由确定性路由分配给专业智能体：

- knowledge_agent：知识库检索与问答，只持有 RAG 工具。
- report_agent：报告分析与图表生成，持有 RAG 和图表工具。
- utility_agent：天气、地图及外部 MCP 工具。
- memory 节点：处理显式的记住、查看和删除请求；不会让模型自行采集用户信息。

LangGraph 使用 Checkpointer 按 `thread_id` 持久化会话状态。Streamlit 另外生成稳定的随机 `user_id`，长期记忆可跨聊天检索，但不同用户严格隔离。RAG 使用向量 + BM25 混合召回，再通过 DashScope `gte-rerank` 重排；重排服务失败时会保留原召回顺序。

## 记忆与隐私

长期记忆默认使用 `.agent_data/long_term_memory.sqlite3`，只接受以下显式指令：

- `请记住我喜欢手冲咖啡`
- `查看记忆`
- `忘记我喜欢手冲咖啡`
- `清除所有记忆`

普通对话不会自动写入长期记忆。记忆默认保留 180 天、每个用户最多 200 条、单条最多 2000 字符；相关记忆只作为不可信事实参考注入模型，不会被当成系统指令。侧边栏需二次确认才会清除当前用户的全部长期记忆。

Demo 通过 URL 中的随机 UUID 维持身份连续性，它是隔离标识而不是认证机制；不要公开分享包含 `user` 参数的地址。正式环境必须使用登录系统提供的不可伪造用户 ID。

## 可观测性

每次请求会生成关联 ID，并写入日志。内置指标记录请求成功、失败与超时数，路由分布，模型 token（供应商返回用量时），以及请求和工具调用延迟。Streamlit 侧边栏的“运行状态”可直接查看当前进程快照。日志不会记录原始工具参数值，并继续执行密钥脱敏和按天轮转。

## 环境要求

- Python 3.11–3.13
- 有效的 DashScope API Key

## 安装

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
~~~

在 .env 中填写：

~~~text
DASHSCOPE_API_KEY=your-key
~~~

如需启用 MCP 服务，复制示例配置：

~~~powershell
Copy-Item mcp_config.example.json mcp_config.json
~~~

配置中的 `${DASHSCOPE_API_KEY}` 会在运行时从环境变量展开。每个 MCP 服务独立加载并设置超时，单个服务不可用时不会影响其他 Agent；可通过 `isActive: false` 禁用服务，或使用 `allowedTools` 限制暴露的工具。

Checkpoint 默认写入 `.agent_data/checkpoints.sqlite3`，路径、30 天保留期和最多 500 个会话均可在 `config/agent.yml` 中调整。浏览器 URL 的随机 `thread` 参数用于跨刷新和服务重启恢复页面历史。SQLite 适合本地 Demo；多实例部署可通过环境变量切换官方 PostgreSQL Checkpointer：

~~~text
AGENT_CHECKPOINT_BACKEND=postgres
AGENT_CHECKPOINT_DSN=postgresql://agent:password@postgres:5432/agent_demo?sslmode=disable
LANGGRAPH_STRICT_MSGPACK=true
~~~

## 运行

~~~powershell
streamlit run app.py --server.fileWatcherType=none
~~~

## 容器化运行

项目包含非 root 运行的 Dockerfile、Streamlit 健康检查，以及带 PostgreSQL 的 Compose 基线：

~~~powershell
$env:DASHSCOPE_API_KEY="your-key"
$env:POSTGRES_PASSWORD="replace-with-a-strong-password"
$env:AGENT_CHECKPOINT_DSN="postgresql://agent:URL-encoded-password@postgres:5432/agent_demo?sslmode=disable"
docker compose up --build
~~~

服务地址为 `http://localhost:8501`。Compose 使用 PostgreSQL 共享短期会话 Checkpoint，支持横向扩展应用实例；长期记忆的默认 SQLite 文件仍只适合单实例 Demo，生产多实例必须接入共享记忆实现。镜像会携带当前 Chroma 索引，运行数据分别存入命名卷。

## 工程文档

- [架构与信任边界](docs/architecture.md)
- [架构决策 ADR-0001](docs/adr/0001-langgraph-supervisor.md)
- [运维手册](docs/operations.md)
- [故障恢复说明](docs/disaster-recovery.md)

## 验证

~~~powershell
python -m pytest -W error -p no:cacheprovider -m "not e2e"
python -m ruff check .
python -m ruff format --check app.py healthcheck.py agent model rag utils tests
python -m mypy
python -m compileall -q app.py healthcheck.py agent rag model utils
~~~

测试默认使用假智能体，不会调用模型、知识库或远程 MCP 服务。浏览器 E2E 由 CI 启动 Streamlit 与 Chromium 后单独执行。
