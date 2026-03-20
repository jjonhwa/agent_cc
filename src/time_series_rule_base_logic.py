# Re-export from rule_base package for backward compatibility.
from src.rule_base import RULE_REGISTRY
from src.rule_base.common import normalize_signals

__all__ = ["RULE_REGISTRY", "normalize_signals"]
