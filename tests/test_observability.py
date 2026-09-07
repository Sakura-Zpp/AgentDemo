import logging
import unittest
from unittest.mock import patch

from agent.checkpoint import CheckpointProvider
from utils.logger_handler import RequestContextFilter
from utils.observability import metrics, request_context, request_id_var, timed_metric


class ObservabilityTest(unittest.TestCase):
    def setUp(self):
        metrics.reset()

    def test_request_context_is_added_to_log_record_and_reset(self):
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "ok", (), None)
        with request_context("correlation-123"):
            RequestContextFilter().filter(record)
            self.assertEqual(record.request_id, "correlation-123")
        self.assertEqual(request_id_var.get(), "-")

    def test_metrics_snapshot_contains_counter_and_latency_summary(self):
        metrics.increment("requests.total")
        with timed_metric("requests.latency"):
            pass

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["counters"]["requests.total"], 1)
        self.assertEqual(snapshot["durations_ms"]["requests.latency"]["count"], 1)

    def test_postgres_backend_can_be_selected_from_environment(self):
        with patch.dict(
            "os.environ",
            {
                "AGENT_CHECKPOINT_BACKEND": "postgres",
                "AGENT_CHECKPOINT_DSN": "postgresql://agent:test@db/agent_demo",
            },
        ):
            provider = CheckpointProvider.from_config()

        self.assertEqual(provider.backend, "postgres")
        self.assertEqual(provider.dsn, "postgresql://agent:test@db/agent_demo")


if __name__ == "__main__":
    unittest.main()
