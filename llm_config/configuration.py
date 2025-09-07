# llm_config/configuration.py
"""
Configuration management with persistence and validation.
"""

import json
import os
import logging
from typing import Optional
from .interfaces import IConfigurationManager, LLMConfiguration

logger = logging.getLogger(__name__)


class ConfigurationManager(IConfigurationManager):
    """Manages LLM configuration persistence and defaults."""
    
    def __init__(self, config_file: str = "llm_config.json"):
        self.config_file = config_file
        self.config_dir = os.path.dirname(os.path.abspath(config_file))
        
        # Ensure config directory exists
        if self.config_dir and not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
    
    def save_configuration(self, config: LLMConfiguration) -> bool:
        """Save configuration to persistent storage."""
        try:
            config_dict = {
                'provider': config.provider,
                'model': config.model,
                'api_key': config.api_key,
                'temperature': config.temperature,
                'max_retries': config.max_retries,
                'helper_model': config.helper_model,
                'use_rag': config.use_rag,
                'use_visual_fix_code': config.use_visual_fix_code,
                'use_context_learning': config.use_context_learning,
                'verbose': config.verbose,
                'max_scene_concurrency': config.max_scene_concurrency
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False
    
    def load_configuration(self) -> Optional[LLMConfiguration]:
        """Load configuration from persistent storage."""
        try:
            if not os.path.exists(self.config_file):
                logger.info(f"Configuration file {self.config_file} not found")
                return None
            
            with open(self.config_file, 'r') as f:
                config_dict = json.load(f)
            
            config = LLMConfiguration(
                provider=config_dict.get('provider', 'OpenAI'),
                model=config_dict.get('model', 'gpt-4'),
                api_key=config_dict.get('api_key', ''),
                temperature=config_dict.get('temperature', 0.7),
                max_retries=config_dict.get('max_retries', 3),
                helper_model=config_dict.get('helper_model'),
                use_rag=config_dict.get('use_rag', True),
                use_visual_fix_code=config_dict.get('use_visual_fix_code', False),
                use_context_learning=config_dict.get('use_context_learning', True),
                verbose=config_dict.get('verbose', False),
                max_scene_concurrency=config_dict.get('max_scene_concurrency', 1)
            )
            
            logger.info(f"Configuration loaded from {self.config_file}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return None
    
    def get_default_configuration(self) -> LLMConfiguration:
        """Get default configuration."""
        return LLMConfiguration(
            provider='OpenAI',
            model='gpt-4',
            api_key='',
            temperature=0.7,
            max_retries=3,
            # Use a stable helper model or leave None so it mirrors planner
            helper_model=None,
            use_rag=True,
            use_visual_fix_code=False,
            use_context_learning=True,
            verbose=False,
            max_scene_concurrency=1
        )
    
    def backup_configuration(self) -> bool:
        """Create a backup of the current configuration."""
        try:
            if not os.path.exists(self.config_file):
                return True  # Nothing to backup
            
            backup_file = f"{self.config_file}.backup"
            
            with open(self.config_file, 'r') as src:
                with open(backup_file, 'w') as dst:
                    dst.write(src.read())
            
            logger.info(f"Configuration backed up to {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup configuration: {str(e)}")
            return False
    
    def restore_configuration(self) -> Optional[LLMConfiguration]:
        """Restore configuration from backup."""
        try:
            backup_file = f"{self.config_file}.backup"
            
            if not os.path.exists(backup_file):
                logger.warning("No backup file found")
                return None
            
            # Replace current config with backup
            with open(backup_file, 'r') as src:
                with open(self.config_file, 'w') as dst:
                    dst.write(src.read())
            
            logger.info("Configuration restored from backup")
            return self.load_configuration()
            
        except Exception as e:
            logger.error(f"Failed to restore configuration: {str(e)}")
            return None
