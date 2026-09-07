# AgentDemo 架构说明

## 请求链路

1. Streamlit 为浏览器生成独立的 `thread_id` 与 `user_id`。
2. `MultiAgentSystem` 建立请求关联 ID、超时边界和 Checkpoint 上下文。
3. LangGraph Supervisor 以确定性规则选择 `knowledge`、`report`、`utility` 或 `memory`。
4. 工作智能体只能访问职责范围内的工具；最终只向页面返回最后一条 AI 消息。
5. 短期会话写入 LangGraph Checkpointer，显式长期记忆写入独立存储。

## 组件边界

- `agent/router.py`：纯路由策略，不调用模型或外部服务。
- `agent/workers.py`：创建三个工具权限隔离的模型智能体。
- `agent/multi_agent.py`：编排、超时、错误持久化和输出过滤。
- `agent/checkpoint.py`：SQLite/PostgreSQL 会话状态适配与保留策略。
- `agent/memory.py`：用户主动授权的长期记忆、隔离、检索和删除。
- `agent/mcp/`：外部 MCP 配置、超时、白名单和图表适配。
- `rag/`：Chroma + BM25 召回、DashScope 重排和索引生命周期。
- `utils/`：配置校验、日志脱敏和可观测性。

## 状态与信任边界

- `thread_id` 仅隔离会话，`user_id` 仅隔离记忆；二者不是认证凭证。
- 长期记忆只有明确的“记住”命令才能写入，注入模型时作为不可信事实而非指令。
- MCP 配置中的 `${ENV_NAME}` 只从环境变量展开；日志不记录参数值。
- 图表输入必须包含声明的横轴和有限数值，非法数据会失败而不会被改成零。

## 部署形态

- 本地单实例：SQLite Checkpoint + SQLite 长期记忆。
- 多实例基线：PostgreSQL Checkpoint；长期记忆仍需部署方替换为共享存储后才能保证跨实例一致。
- RAG 索引随镜像或共享卷交付；索引版本文件触发进程内检索器重建。

关键决策见 [ADR-0001](adr/0001-langgraph-supervisor.md)。
