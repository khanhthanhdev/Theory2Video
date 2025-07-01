# llm_config/ui_manager.py
"""
UI state management with clean separation of concerns.
"""

import gradio as gr
import logging
from typing import Dict, List, Optional, Tuple, Any
from .interfaces import IUIStateManager, APIKeyValidationResult, IProviderManager, INotificationService

logger = logging.getLogger(__name__)


class UIStateManager(IUIStateManager):
    """Manages UI state changes and updates for LLM configuration."""
    
    def __init__(self, provider_manager: IProviderManager, notification_service: INotificationService):
        self.provider_manager = provider_manager
        self.notification_service = notification_service
        self.current_provider: Optional[str] = None
        self.current_model: Optional[str] = None
        
    def update_provider_selection(self, provider: str) -> Dict:
        """Update UI when provider is selected."""
        try:
            if not self.provider_manager.is_valid_provider(provider):
                logger.warning(f"Invalid provider selected: {provider}")
                return self._create_error_update("Invalid provider selected")
            
            self.current_provider = provider
            models = self.provider_manager.get_models(provider)
            default_model = models[0] if models else None
            self.current_model = default_model
            
            # Check if API key is already set
            has_key = self.provider_manager.has_api_key(provider)
            api_key_value = self.provider_manager.get_api_key(provider) if has_key else ""
            
            # Get provider description
            provider_desc = self.provider_manager.get_provider_description(provider)
            
            return {
                'model_dropdown': gr.update(
                    choices=models,
                    value=default_model,
                    visible=len(models) > 0
                ),
                'api_key_input': gr.update(
                    value=api_key_value,
                    placeholder=f"Enter your {provider} API key"
                ),
                'provider_info': gr.update(
                    value=f"**{provider}**: {provider_desc}",
                    visible=True
                ),
                'model_description': gr.update(
                    value=self.provider_manager.get_model_description(default_model) if default_model else "",
                    visible=default_model is not None
                ),
                'validation_status': gr.update(
                    value="✅ API key found" if has_key else "⚠️ API key required",
                    visible=True
                )
            }
            
        except Exception as e:
            logger.error(f"Error updating provider selection: {str(e)}")
            return self._create_error_update(f"Error updating provider: {str(e)}")
    
    def update_model_selection(self, model: str) -> Dict:
        """Update UI when model is selected."""
        try:
            if not model:
                return {}
            
            self.current_model = model
            model_description = self.provider_manager.get_model_description(model)
            
            return {
                'model_description': gr.update(
                    value=model_description,
                    visible=True
                )
            }
            
        except Exception as e:
            logger.error(f"Error updating model selection: {str(e)}")
            return self._create_error_update(f"Error updating model: {str(e)}")
    
    def show_validation_feedback(self, result: APIKeyValidationResult) -> Dict:
        """Show validation feedback to user."""
        try:
            if result.is_valid:
                status_text = "✅ API key is valid"
                status_color = "green"
                if result.error_message:
                    status_text += f" ({result.error_message})"
            else:
                status_text = f"❌ {result.error_message or 'Invalid API key'}"
                status_color = "red"
            
            return {
                'validation_status': gr.update(
                    value=status_text,
                    visible=True
                ),
                'api_key_feedback': gr.update(
                    value=status_text,
                    visible=True
                )
            }
            
        except Exception as e:
            logger.error(f"Error showing validation feedback: {str(e)}")
            return self._create_error_update(f"Error showing feedback: {str(e)}")
    
    def reset_form(self) -> Dict:
        """Reset form to default state."""
        try:
            providers = self.provider_manager.get_providers()
            default_provider = providers[0] if providers else None
            
            if default_provider:
                models = self.provider_manager.get_models(default_provider)
                default_model = models[0] if models else None
            else:
                models = []
                default_model = None
            
            self.current_provider = default_provider
            self.current_model = default_model
            
            return {
                'provider_dropdown': gr.update(
                    value=default_provider,
                    choices=providers
                ),
                'model_dropdown': gr.update(
                    value=default_model,
                    choices=models
                ),
                'api_key_input': gr.update(
                    value="",
                    placeholder="Enter your API key"
                ),
                'temperature_slider': gr.update(value=0.7),
                'max_retries_slider': gr.update(value=3),
                'max_scene_concurrency_slider': gr.update(value=1),
                'use_rag_checkbox': gr.update(value=True),
                'use_visual_fix_code_checkbox': gr.update(value=False),
                'use_context_learning_checkbox': gr.update(value=True),
                'verbose_checkbox': gr.update(value=False),
                'validation_status': gr.update(
                    value="⚠️ Configuration reset",
                    visible=True
                ),
                'model_description': gr.update(
                    value=self.provider_manager.get_model_description(default_model) if default_model else "",
                    visible=default_model is not None
                )
            }
            
        except Exception as e:
            logger.error(f"Error resetting form: {str(e)}")
            return self._create_error_update(f"Error resetting form: {str(e)}")
    
    def update_helper_model_selection(self, helper_model: str) -> Dict:
        """Update UI when helper model is selected."""
        try:
            if not helper_model:
                return {}
            
            helper_description = self.provider_manager.get_model_description(helper_model)
            
            return {
                'helper_model_description': gr.update(
                    value=f"Helper: {helper_description}",
                    visible=True
                )
            }
            
        except Exception as e:
            logger.error(f"Error updating helper model selection: {str(e)}")
            return {}
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """Get current UI configuration state."""
        return {
            'provider': self.current_provider,
            'model': self.current_model,
            'has_api_key': self.provider_manager.has_api_key(self.current_provider) if self.current_provider else False
        }
    
    def validate_current_configuration(self) -> Tuple[bool, str]:
        """Validate current configuration."""
        if not self.current_provider:
            return False, "No provider selected"
        
        if not self.current_model:
            return False, "No model selected"
        
        if not self.provider_manager.has_api_key(self.current_provider):
            return False, "API key not set for selected provider"
        
        return True, "Configuration is valid"
    
    def _create_error_update(self, error_message: str) -> Dict:
        """Create an error update for UI components."""
        return {
            'validation_status': gr.update(
                value=f"❌ {error_message}",
                visible=True
            )
        }
    
    def show_configuration_summary(self) -> Dict:
        """Show a summary of current configuration."""
        try:
            config = self.get_current_configuration()
            summary_text = f"""
            **Current Configuration:**
            - Provider: {config['provider'] or 'None'}
            - Model: {config['model'] or 'None'}
            - API Key: {'✅ Set' if config['has_api_key'] else '❌ Not set'}
            """
            
            return {
                'configuration_summary': gr.update(
                    value=summary_text,
                    visible=True
                )
            }
            
        except Exception as e:
            logger.error(f"Error showing configuration summary: {str(e)}")
            return {}
