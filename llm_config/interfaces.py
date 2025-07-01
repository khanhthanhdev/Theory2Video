# llm_config/interfaces.py
"""
Abstract interfaces for LLM configuration system following SOLID principles.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class APIKeyValidationResult:
    """Result of API key validation."""
    is_valid: bool
    error_message: Optional[str] = None
    provider_name: Optional[str] = None


@dataclass
class LLMConfiguration:
    """Complete LLM configuration."""
    provider: str
    model: str
    api_key: str
    temperature: float = 0.7
    max_retries: int = 3
    helper_model: Optional[str] = None
    use_rag: bool = True
    use_visual_fix_code: bool = False
    use_context_learning: bool = True
    verbose: bool = False
    max_scene_concurrency: int = 1


class IProviderValidator(ABC):
    """Interface for validating provider configurations."""
    
    @abstractmethod
    async def validate_api_key(self, provider: str, api_key: str) -> APIKeyValidationResult:
        """Validate an API key for a specific provider."""
        pass
    
    @abstractmethod
    def get_supported_providers(self) -> List[str]:
        """Get list of supported providers."""
        pass


class IConfigurationManager(ABC):
    """Interface for managing LLM configurations."""
    
    @abstractmethod
    def save_configuration(self, config: LLMConfiguration) -> bool:
        """Save configuration to persistent storage."""
        pass
    
    @abstractmethod
    def load_configuration(self) -> Optional[LLMConfiguration]:
        """Load configuration from persistent storage."""
        pass
    
    @abstractmethod
    def get_default_configuration(self) -> LLMConfiguration:
        """Get default configuration."""
        pass


class IProviderManager(ABC):
    """Interface for managing providers and models."""
    
    @abstractmethod
    def get_providers(self) -> List[str]:
        """Get available providers."""
        pass
    
    @abstractmethod
    def get_models(self, provider: str) -> List[str]:
        """Get available models for a provider."""
        pass
    
    @abstractmethod
    def get_model_description(self, model: str) -> str:
        """Get description for a model."""
        pass
    
    @abstractmethod
    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider."""
        pass


class IUIStateManager(ABC):
    """Interface for managing UI state."""
    
    @abstractmethod
    def update_provider_selection(self, provider: str) -> Dict:
        """Update UI when provider is selected."""
        pass
    
    @abstractmethod
    def update_model_selection(self, model: str) -> Dict:
        """Update UI when model is selected."""
        pass
    
    @abstractmethod
    def show_validation_feedback(self, result: APIKeyValidationResult) -> Dict:
        """Show validation feedback to user."""
        pass
    
    @abstractmethod
    def reset_form(self) -> Dict:
        """Reset form to default state."""
        pass


class INotificationService(ABC):
    """Interface for user notifications."""
    
    @abstractmethod
    def show_success(self, message: str) -> None:
        """Show success notification."""
        pass
    
    @abstractmethod
    def show_error(self, message: str) -> None:
        """Show error notification."""
        pass
    
    @abstractmethod
    def show_warning(self, message: str) -> None:
        """Show warning notification."""
        pass
