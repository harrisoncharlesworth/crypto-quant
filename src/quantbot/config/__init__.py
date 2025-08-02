import os
import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable overrides."""
    if config_path is None:
        config_path = Path(__file__).parent / "default.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with environment variables
    if os.getenv("DRY_RUN"):
        config["trading"]["dry_run"] = os.getenv("DRY_RUN").lower() == "true"

    if os.getenv("MAX_POSITION_SIZE"):
        config["trading"]["max_position_size"] = float(os.getenv("MAX_POSITION_SIZE"))

    if os.getenv("RISK_LIMIT"):
        config["trading"]["risk_limit"] = float(os.getenv("RISK_LIMIT"))

    return config
