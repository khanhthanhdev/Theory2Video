# provider.py
"""
Provider management for Theory2Manim Gradio app.
Allows user to select provider, enter API key, and select model.
"""

import os
from typing import Dict, List, Optional
from src.utils.model_registry import get_providers_config

class ProviderManager:
    def __init__(self):
        # Load from central registry
        providers_cfg = get_providers_config()
        # Normalize to human-readable keys matching legacy UI labels
        self.providers = {}
        for key, cfg in providers_cfg.items():
            display = cfg.get('display_name', key.title())
            self.providers[display] = {
                'api_key_env': cfg.get('api_key_env', ''),
                'models': cfg.get('models', [])
            }
        self.selected_provider = None
        self.api_keys = {}

    def get_providers(self) -> List[str]:
        return list(self.providers.keys())

    def get_models(self, provider: str) -> List[str]:
        return self.providers.get(provider, {}).get('models', [])

    def set_api_key(self, provider: str, api_key: str):
        env_var = self.providers[provider]['api_key_env']
        os.environ[env_var] = api_key
        self.api_keys[provider] = api_key

    def get_api_key(self, provider: str) -> Optional[str]:
        env_var = self.providers[provider]['api_key_env']
        return os.environ.get(env_var)

    def get_selected_provider(self) -> Optional[str]:
        return self.selected_provider

    def set_selected_provider(self, provider: str):
        self.selected_provider = provider

    def get_selected_model(self) -> Optional[str]:
        if self.selected_provider:
            return self.get_models(self.selected_provider)[0]
        return None

provider_manager = ProviderManager()
