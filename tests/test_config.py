from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
