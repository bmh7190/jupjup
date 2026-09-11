from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jupjup.api_client import Lost112ApiClient
from jupjup.config import Settings


class SettingsTest(unittest.TestCase):
    def test_explicit_env_file_overrides_stale_shell_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "DATA_GO_KR_SERVICE_KEY=data-key\n"
                "OPENAI_API_KEY=project-key\n"
                "OPENAI_MODEL=gpt-5.4-mini\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OPENAI_API_KEY": "stale-shell-key"}):
                settings = Settings.from_env(env_path)

            self.assertEqual(settings.openai_api_key, "project-key")
            self.assertEqual(settings.openai_model, "gpt-5.4-mini")

    def test_search_limits_from_env_are_applied_to_api_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "DATA_GO_KR_SERVICE_KEY=data-key\n"
                "LOST112_PAGE_SIZE=10\n"
                "LOST112_TIMEOUT_SECONDS=60\n"
                "LOST112_DETAIL_LIMIT=10\n"
                "LOST112_MAX_PAGES_PER_WINDOW=3\n"
                "LOST112_SEARCH_BUDGET_SECONDS=40\n",
                encoding="utf-8",
            )

            settings = Settings.from_env(env_path)
            client = Lost112ApiClient(
                settings.data_service_key,
                **settings.api_client_options,
            )

            self.assertEqual(client.page_size, 10)
            self.assertEqual(client.timeout_seconds, 60)
            self.assertEqual(client.detail_limit, 10)
            self.assertEqual(client.max_pages_per_window, 3)
            self.assertEqual(client.search_budget_seconds, 40)


if __name__ == "__main__":
    unittest.main()
