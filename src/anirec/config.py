"""Configuration loader for anirec.

Reads a YAML config file and returns it as a plain dictionary.
All other modules pull paths, thresholds, and hyperparameters from here
instead of hardcoding them.
"""

from pathlib import Path

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load and return the YAML configuration as a dictionary.

    Parameters
    ----------
    path : str | Path | None
        Path to a YAML config file.  Falls back to ``configs/default.yaml``
        when *None*.

    Returns
    -------
    dict
        Parsed configuration.
    """
    path = Path(path) if path else _DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)
