# llm_config/provider_manager.py
"""
Enhanced provider manager with better separation of concerns.
"""

import os
import logging
from typing import Dict, List, Optional
from .interfaces import IProviderManager
from src.utils.model_registry import get_providers_config, get_model_descriptions

logger = logging.getLogger(__name__)


class EnhancedProviderManager(IProviderManager):
    """Enhanced provider manager with better organization and extensibility."""
    
    def __init__(self):
        # Load providers and model descriptions from central registry
        providers_cfg = get_providers_config()
        self.providers_config = {}
        for key, cfg in providers_cfg.items():
            display = cfg.get('display_name', key.title())
            self.providers_config[display] = {
                'api_key_env': cfg.get('api_key_env', ''),
                'models': cfg.get('models', []),
                'display_name': display,
                'description': cfg.get('description', '')
            }

        self.model_descriptions = get_model_descriptions()
        
        self.api_keys: Dict[str, str] = {}
        self.selected_provider: Optional[str] = None
        self.selected_model: Optional[str] = None
    
    def get_providers(self) -> List[str]:
        """Get available providers."""
        return list(self.providers_config.keys())
    
    def get_models(self, provider: str) -> List[str]:
        """Get available models for a provider."""
        if provider not in self.providers_config:
            logger.warning(f"Provider '{provider}' not found")
            return []
        
        return self.providers_config[provider].get('models', [])
    
    def get_model_description(self, model: str) -> str:
        """Get description for a model."""
        return self.model_descriptions.get(model, "No description available")
    
    def get_provider_description(self, provider: str) -> str:
        """Get description for a provider."""
        if provider not in self.providers_config:
            return "Unknown provider"
        
        return self.providers_config[provider].get('description', 'No description available')
    
    def get_provider_display_name(self, provider: str) -> str:
        """Get display name for a provider."""
        if provider not in self.providers_config:
            return provider
        
        return self.providers_config[provider].get('display_name', provider)
    
    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider."""
        if provider not in self.providers_config:
            logger.error(f"Cannot set API key for unknown provider: {provider}")
            return
        
        env_var = self.providers_config[provider]['api_key_env']
        os.environ[env_var] = api_key
        self.api_keys[provider] = api_key
        
        logger.info(f"API key set for provider: {provider}")
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        if provider not in self.providers_config:
            return None
        
        # First check our internal storage
        if provider in self.api_keys:
            return self.api_keys[provider]
        
        # Then check environment variable
        env_var = self.providers_config[provider]['api_key_env']
        return os.environ.get(env_var)
    
    def has_api_key(self, provider: str) -> bool:
        """Check if provider has an API key set."""
        api_key = self.get_api_key(provider)
        return api_key is not None and api_key.strip() != ""
    
    def get_default_model(self, provider: str) -> Optional[str]:
        """Get the default (first) model for a provider."""
        models = self.get_models(provider)
        return models[0] if models else None
    
    def is_valid_provider(self, provider: str) -> bool:
        """Check if provider is valid."""
        return provider in self.providers_config
    
    def is_valid_model(self, provider: str, model: str) -> bool:
        """Check if model is valid for the given provider."""
        return model in self.get_models(provider)
    
    def get_models_with_descriptions(self, provider: str) -> Dict[str, str]:
        """Get models with their descriptions for a provider."""
        models = self.get_models(provider)
        return {model: self.get_model_description(model) for model in models}
    
    def clear_api_key(self, provider: str) -> None:
        """Clear API key for a provider."""
        if provider in self.api_keys:
            del self.api_keys[provider]
        
        if provider in self.providers_config:
            env_var = self.providers_config[provider]['api_key_env']
            if env_var in os.environ:
                del os.environ[env_var]
        
        logger.info(f"API key cleared for provider: {provider}")
    
    def get_configuration_summary(self) -> Dict:
        """Get a summary of current configuration."""
        return {
            'total_providers': len(self.providers_config),
            'providers_with_keys': sum(1 for p in self.providers_config if self.has_api_key(p)),
            'selected_provider': self.selected_provider,
            'selected_model': self.selected_model,
            'available_models': sum(len(config['models']) for config in self.providers_config.values())
        }
