import tempfile
import unittest
from pathlib import Path

from utils.config import load_agent_config, load_mcp_config, load_rag_config


class McpConfigTest(unittest.TestCase):
    def test_missing_mcp_config_is_optional(self):
        missing_path = Path(tempfile.gettempdir()) / "agent-demo-missing-mcp.json"
        self.assertEqual(
            load_mcp_config(str(missing_path)),
            {"mcpServers": {}},
        )

    def test_rag_config_rejects_top_n_greater_than_candidate_k(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            config_path = Path(temp_directory) / "rag.yml"
            config_path.write_text(
                """
chat_model_name: qwen-plus
embedding_model_name: text-embedding-v4
rerank_enabled: true
rerank_model_name: gte-rerank
rerank_candidate_k: 2
rerank_top_n: 3
""".strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "不能大于"):
                load_rag_config(config_path)

    def test_agent_config_rejects_unknown_checkpoint_backend(self):
        source = Path("config/agent.yml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp_directory:
            config_path = Path(temp_directory) / "agent.yml"
            config_path.write_text(
                source.replace("backend: sqlite", "backend: unknown"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "仅支持"):
                load_agent_config(config_path)


if __name__ == "__main__":
    unittest.main()
