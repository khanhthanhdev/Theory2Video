import json
import os
from pathlib import Path
from typing import Dict, List, Optional


_REGISTRY_PATH = Path(__file__).with_name("allowed_models.json")


def _load_registry() -> Dict:
    try:
        with _REGISTRY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_allowed_models() -> List[str]:
    data = _load_registry()
    models = data.get("allowed_models", [])
    # ensure unique and preserve order
    seen = set()
    out = []
    for m in models:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def get_default_model() -> str:
    data = _load_registry()
    models = get_allowed_models()
    return data.get("default_model") or (models[0] if models else "openai/gpt-4o")


def get_providers_config() -> Dict[str, Dict]:
    data = _load_registry()
    return data.get("providers", {})


def get_model_descriptions() -> Dict[str, str]:
    data = _load_registry()
    return data.get("model_descriptions", {})


def get_embedding_models() -> List[str]:
    data = _load_registry()
    return data.get("embedding_models", [])

