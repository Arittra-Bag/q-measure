import json
from pathlib import Path
from typing import Dict, Any

def save_results(out_dir: str, payload: Dict[str, Any]) -> str:
    """Save JSON + derived CSV summaries. Return path."""

def make_markdown_report(payload: Dict[str, Any]) -> str:
    """Return markdown string summarizing config, key plots pointers, metrics."""
