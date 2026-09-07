import logging
import unittest

from utils.logger_handler import RedactingFormatter, redact_sensitive


class LogRedactionTest(unittest.TestCase):
    def test_redacts_bearer_and_api_key(self):
        message = "Authorization: Bearer abc.def-123 api_key=private-value"
        redacted = redact_sensitive(message)

        self.assertNotIn("abc.def-123", redacted)
        self.assertNotIn("private-value", redacted)
        self.assertIn("Authorization: ***", redacted)

    def test_formatter_redacts_exception_text(self):
        formatter = RedactingFormatter("%(message)s")
        try:
            raise RuntimeError("token sk-secretvalue123")
        except RuntimeError:
            record = logging.LogRecord(
                "test",
                logging.ERROR,
                __file__,
                1,
                "failed",
                (),
                exc_info=__import__("sys").exc_info(),
            )

        formatted = formatter.format(record)
        self.assertNotIn("sk-secretvalue123", formatted)
        self.assertIn("sk-***", formatted)
