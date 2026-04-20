import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


LOCAL_CONFIG_ENV_VAR = "TRADINGAGENTS_LOCAL_CONFIG"
DEFAULT_LOCAL_CONFIG_FILENAME = ".tradingagents.local.json"


def get_local_config_path(project_root: Path) -> Path:
    """Return the local CLI config path, honoring an env override when present."""
    override = os.getenv(LOCAL_CONFIG_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return project_root / DEFAULT_LOCAL_CONFIG_FILENAME


def load_local_config(project_root: Path) -> Tuple[Dict[str, Any], Path]:
    """Load local CLI defaults from JSON. Missing files return an empty config."""
    config_path = get_local_config_path(project_root)
    if not config_path.exists():
        return {}, config_path

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Local config file must contain a JSON object: {config_path}"
        )

    return data, config_path
