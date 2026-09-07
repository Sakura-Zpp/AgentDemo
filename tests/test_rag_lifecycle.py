import unittest
from unittest.mock import patch

from agent.tools.tools import get_rag_search, reset_rag_search


class RagLifecycleTest(unittest.TestCase):
    def tearDown(self):
        reset_rag_search()

    def test_rag_service_is_lazily_reused(self):
        service = object()
        with patch("agent.tools.tools.RagSearch", return_value=service) as rag_class:
            first = get_rag_search()
            second = get_rag_search()

        self.assertIs(first, service)
        self.assertIs(second, service)
        rag_class.assert_called_once_with()

    def test_rag_service_rebuilds_when_index_revision_changes(self):
        first_service = object()
        second_service = object()
        with (
            patch("agent.tools.tools.get_index_revision", side_effect=[1, 1, 2]),
            patch(
                "agent.tools.tools.RagSearch",
                side_effect=[first_service, second_service],
            ) as rag_class,
        ):
            first = get_rag_search()
            repeated = get_rag_search()
            refreshed = get_rag_search()

        self.assertIs(first, first_service)
        self.assertIs(repeated, first_service)
        self.assertIs(refreshed, second_service)
        self.assertEqual(rag_class.call_count, 2)
