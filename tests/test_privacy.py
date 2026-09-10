from __future__ import annotations

import unittest

from lost112_agent.privacy import mask_pii


class PrivacyTest(unittest.TestCase):
    def test_sensitive_patterns_are_masked(self) -> None:
        text = "연락처 010-1234-5678, 카드 1234-5678-9012-3456"
        masked = mask_pii(text)
        self.assertNotIn("010-1234-5678", masked)
        self.assertNotIn("1234-5678-9012-3456", masked)


if __name__ == "__main__":
    unittest.main()

