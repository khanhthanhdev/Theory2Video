# provider.py
"""
Provider management for Theory2Manim Gradio app.
Allows user to select provider, enter API key, and select model.
"""

import os
from typing import Dict, List, Optional

class ProviderManager:
    def __init__(self):
        # Example provider configs; extend as needed
        self.providers = {
            'OpenAI': {
                'api_key_env': 'OPENAI_API_KEY',
                'models': [
                    'gpt-4.1', 'gpt-4o', 'gpt-3.5-turbo'
                ]
            },
            'Google Gemini': {
                'api_key_env': 'GOOGLE_API_KEY',
                'models': [
                    'gemini-1.5-pro-002', 'gemini-2.5-flash-preview-04-17'
                ]
            },
            'Anthropic': {
                'api_key_env': 'ANTHROPIC_API_KEY',
                'models': [
                    'claude-3-5-sonnet-20241022', 'claude-3-haiku'
                ]
            },
            'OpenRouter': {
                'api_key_env': 'OPENROUTER_API_KEY',
                'models': [
                    'openai/gpt-4o', 'openai/gpt-4o-mini', 'anthropic/claude-3.5-sonnet',
                    'anthropic/claude-3-haiku', 'google/gemini-pro-1.5', 'deepseek/deepseek-chat',
                    'qwen/qwen-2.5-72b-instruct', 'meta-llama/llama-3.1-8b-instruct:free',
                    'microsoft/phi-3-mini-128k-instruct:free'
                ]
            }
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
