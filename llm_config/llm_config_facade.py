# llm_config/llm_config_facade.py
"""
Facade pattern implementation for clean LLM configuration management.
This provides a simple interface to the complex subsystem.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from .interfaces import (
    LLMConfiguration, APIKeyValidationResult,
    IProviderValidator, IConfigurationManager, IProviderManager, 
    IUIStateManager, INotificationService
)
from .validation import ProviderValidator
from .configuration import ConfigurationManager
from .provider_manager import EnhancedProviderManager
from .ui_manager import UIStateManager
from .notifications import GradioNotificationService

logger = logging.getLogger(__name__)


class LLMConfigurationFacade:
    """
    Facade for LLM configuration system.
    Provides a simple interface to manage providers, models, and API keys.
    """
    
    def __init__(self, config_file: str = "llm_config.json"):
        # Initialize all components
        self.validator: IProviderValidator = ProviderValidator()
        self.config_manager: IConfigurationManager = ConfigurationManager(config_file)
        self.provider_manager: IProviderManager = EnhancedProviderManager()
        self.notification_service: INotificationService = GradioNotificationService()
        self.ui_manager: IUIStateManager = UIStateManager(
            self.provider_manager, 
            self.notification_service
        )
        
        # Load saved configuration if available
        self._load_saved_configuration()
    
    def _load_saved_configuration(self) -> None:
        """Load previously saved configuration."""
        try:
            config = self.config_manager.load_configuration()
            if config:
                # Set API key in provider manager
                if config.api_key:
                    self.provider_manager.set_api_key(config.provider, config.api_key)
                
                logger.info("Loaded saved configuration")
            else:
                logger.info("No saved configuration found, using defaults")
        except Exception as e:
            logger.error(f"Failed to load saved configuration: {str(e)}")
    
    # Provider Management
    def get_providers(self) -> List[str]:
        """Get list of available providers."""
        return self.provider_manager.get_providers()
    
    def get_models(self, provider: str) -> List[str]:
        """Get models for a specific provider."""
        return self.provider_manager.get_models(provider)
    
    def get_model_description(self, model: str) -> str:
        """Get description for a model."""
        return self.provider_manager.get_model_description(model)
    
    def get_provider_description(self, provider: str) -> str:
        """Get description for a provider."""
        return self.provider_manager.get_provider_description(provider)
    
    # API Key Management
    def set_api_key(self, provider: str, api_key: str) -> bool:
        """Set API key for a provider."""
        try:
            if not api_key or not api_key.strip():
                self.notification_service.show_error("API key cannot be empty")
                return False
            
            self.provider_manager.set_api_key(provider, api_key)
            self.notification_service.show_success(f"API key set for {provider}")
            return True
        except Exception as e:
            logger.error(f"Failed to set API key: {str(e)}")
            self.notification_service.show_error(f"Failed to set API key: {str(e)}")
            return False
    
    def validate_api_key(self, provider: str, api_key: str) -> APIKeyValidationResult:
        """Validate an API key synchronously."""
        return self.validator.validate_api_key_sync(provider, api_key)
    
    async def validate_api_key_async(self, provider: str, api_key: str) -> APIKeyValidationResult:
        """Validate an API key asynchronously."""
        return await self.validator.validate_api_key(provider, api_key)
    
    def has_api_key(self, provider: str) -> bool:
        """Check if provider has an API key set."""
        return self.provider_manager.has_api_key(provider)
    
    # Configuration Management
    def save_configuration(self, config: LLMConfiguration) -> bool:
        """Save complete configuration."""
        try:
            success = self.config_manager.save_configuration(config)
            if success:
                self.notification_service.show_success("Configuration saved successfully")
            else:
                self.notification_service.show_error("Failed to save configuration")
            return success
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            self.notification_service.show_error(f"Failed to save configuration: {str(e)}")
            return False
    
    def load_configuration(self) -> Optional[LLMConfiguration]:
        """Load saved configuration."""
        return self.config_manager.load_configuration()
    
    def get_default_configuration(self) -> LLMConfiguration:
        """Get default configuration."""
        return self.config_manager.get_default_configuration()
    
    def create_configuration(self, provider: str, model: str, api_key: str, **kwargs) -> LLMConfiguration:
        """Create a new LLM configuration."""
        return LLMConfiguration(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7),
            max_retries=kwargs.get('max_retries', 3),
            helper_model=kwargs.get('helper_model'),
            use_rag=kwargs.get('use_rag', True),
            use_visual_fix_code=kwargs.get('use_visual_fix_code', False),
            use_context_learning=kwargs.get('use_context_learning', True),
            verbose=kwargs.get('verbose', False),
            max_scene_concurrency=kwargs.get('max_scene_concurrency', 1)
        )
    
    # UI State Management
    def update_provider_selection(self, provider: str) -> Dict:
        """Update UI when provider is selected."""
        return self.ui_manager.update_provider_selection(provider)
    
    def update_model_selection(self, model: str) -> Dict:
        """Update UI when model is selected."""
        return self.ui_manager.update_model_selection(model)
    
    def show_validation_feedback(self, result: APIKeyValidationResult) -> Dict:
        """Show validation feedback in UI."""
        return self.ui_manager.show_validation_feedback(result)
    
    def reset_form(self) -> Dict:
        """Reset form to default state."""
        return self.ui_manager.reset_form()
    
    def get_current_ui_state(self) -> Dict[str, Any]:
        """Get current UI state."""
        return self.ui_manager.get_current_configuration()
    
    def validate_current_configuration(self) -> Tuple[bool, str]:
        """Validate current configuration."""
        return self.ui_manager.validate_current_configuration()
    
    # Utility Methods
    def get_configuration_summary(self) -> Dict:
        """Get summary of current configuration."""
        return self.provider_manager.get_configuration_summary()
    
    def test_configuration(self, config: LLMConfiguration) -> Tuple[bool, str]:
        """Test a configuration by validating API key."""
        try:
            result = self.validate_api_key(config.provider, config.api_key)
            return result.is_valid, result.error_message or "Configuration test completed"
        except Exception as e:
            logger.error(f"Failed to test configuration: {str(e)}")
            return False, f"Test failed: {str(e)}"
    
    def backup_configuration(self) -> bool:
        """Create a backup of current configuration."""
        return self.config_manager.backup_configuration()
    
    def restore_configuration(self) -> Optional[LLMConfiguration]:
        """Restore configuration from backup."""
        config = self.config_manager.restore_configuration()
        if config:
            self.notification_service.show_success("Configuration restored from backup")
        else:
            self.notification_service.show_error("Failed to restore configuration")
        return config
    
    def clear_all_api_keys(self) -> None:
        """Clear all stored API keys."""
        try:
            for provider in self.get_providers():
                self.provider_manager.clear_api_key(provider)
            self.notification_service.show_success("All API keys cleared")
        except Exception as e:
            logger.error(f"Failed to clear API keys: {str(e)}")
            self.notification_service.show_error(f"Failed to clear API keys: {str(e)}")
    
    def get_last_notification(self) -> Optional[str]:
        """Get last notification message."""
        return self.notification_service.get_last_notification()
    
    def initialize_ui_defaults(self) -> Dict:
        """Initialize UI with default values."""
        try:
            providers = self.get_providers()
            if not providers:
                return {}
            
            default_provider = providers[0]
            models = self.get_models(default_provider)
            default_model = models[0] if models else None
            
            return {
                'provider_choices': providers,
                'provider_value': default_provider,
                'model_choices': models,
                'model_value': default_model,
                'model_description': self.get_model_description(default_model) if default_model else "",
                'has_api_key': self.has_api_key(default_provider)
            }
        except Exception as e:
            logger.error(f"Failed to initialize UI defaults: {str(e)}")
            return {}
