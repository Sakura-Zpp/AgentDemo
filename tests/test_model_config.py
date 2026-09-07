import importlib
import os
import unittest
from unittest.mock import patch

import model.factory


class ModelEnvironmentTest(unittest.TestCase):
    def test_process_environment_is_not_overridden_by_dotenv(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "runtime-sentinel"}):
            reloaded = importlib.reload(model.factory)

        self.assertEqual(reloaded.DASHSCOPE_API_KEY, "runtime-sentinel")
