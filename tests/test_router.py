import unittest

from agent.router import RoutingPolicy


class RoutingPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = RoutingPolicy(
            report_keywords=("报告", "图表"),
            utility_keywords=("天气", "地图"),
        )

    def test_routes_report_request(self):
        self.assertEqual(self.policy.route("生成年度财报图表"), "report")

    def test_routes_utility_request(self):
        self.assertEqual(self.policy.route("查询上海天气"), "utility")

    def test_routes_normal_question_to_knowledge(self):
        self.assertEqual(self.policy.route("公司的行为总则是什么"), "knowledge")

    def test_routes_explicit_memory_command(self):
        self.assertEqual(self.policy.route("请记住我喜欢咖啡"), "memory")

    def test_memory_keyword_in_normal_sentence_does_not_hijack_route(self):
        self.assertEqual(self.policy.route("系统怎样记住用户偏好"), "knowledge")

    def test_short_follow_up_keeps_previous_agent(self):
        self.assertEqual(self.policy.route("继续", last_agent="report"), "report")

    def test_normal_short_question_is_routed_again(self):
        self.assertEqual(
            self.policy.route("公司制度是什么", last_agent="utility"),
            "knowledge",
        )

    def test_explicit_intent_overrides_previous_agent(self):
        self.assertEqual(
            self.policy.route("查询天气", last_agent="report"),
            "utility",
        )


if __name__ == "__main__":
    unittest.main()
