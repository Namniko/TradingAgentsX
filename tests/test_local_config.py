import json
import shutil
import unittest
import uuid
from pathlib import Path

from cli.local_config import DEFAULT_LOCAL_CONFIG_FILENAME, load_local_config


class LocalConfigTests(unittest.TestCase):
    def _make_workspace_temp_dir(self) -> Path:
        root = Path(__file__).resolve().parent / "_tmp_local_config"
        root.mkdir(exist_ok=True)
        path = root / uuid.uuid4().hex
        path.mkdir()
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_missing_local_config_returns_empty_dict(self):
        project_root = self._make_workspace_temp_dir()
        data, path = load_local_config(project_root)

        self.assertEqual(data, {})
        self.assertEqual(path, project_root / DEFAULT_LOCAL_CONFIG_FILENAME)

    def test_loads_json_object_from_default_path(self):
        project_root = self._make_workspace_temp_dir()
        config_path = project_root / DEFAULT_LOCAL_CONFIG_FILENAME
        config_path.write_text(
            json.dumps(
                {
                    "output_language": "Chinese",
                    "llm_provider": "openai",
                    "quick_think_llm": "gpt-5.4-mini",
                    "deep_think_llm": "gpt-5.4",
                }
            ),
            encoding="utf-8",
        )

        data, path = load_local_config(project_root)

        self.assertEqual(path, config_path)
        self.assertEqual(data["output_language"], "Chinese")
        self.assertEqual(data["quick_think_llm"], "gpt-5.4-mini")


if __name__ == "__main__":
    unittest.main()
