# llm_config/validation.py
"""
Provider validation implementation with proper error handling and async support.
"""

import asyncio
import aiohttp
import os
import logging
from typing import Dict, List, Optional
from .interfaces import IProviderValidator, APIKeyValidationResult

logger = logging.getLogger(__name__)


class ProviderValidator(IProviderValidator):
    """Validates API keys and provider configurations."""
    
    def __init__(self):
        self.validation_endpoints = {
            'OpenAI': {
                'url': 'https://api.openai.com/v1/models',
                'headers_fn': lambda key: {'Authorization': f'Bearer {key}'}
            },
            'Google Gemini': {
                'url': 'https://generativelanguage.googleapis.com/v1beta/models',
                'headers_fn': lambda key: {},
                'params_fn': lambda key: {'key': key}
            },
            'Anthropic': {
                'url': 'https://api.anthropic.com/v1/messages',
                'headers_fn': lambda key: {
                    'x-api-key': key,
                    'anthropic-version': '2023-06-01',
                    'content-type': 'application/json'
                },
                'method': 'POST',
                'data': {
                    'model': 'claude-3-haiku-20240307',
                    'max_tokens': 1,
                    'messages': [{'role': 'user', 'content': 'test'}]
                }
            },
            'OpenRouter': {
                'url': 'https://openrouter.ai/api/v1/models',
                'headers_fn': lambda key: {'Authorization': f'Bearer {key}'}
            }
        }
        
        # Timeout for validation requests
        self.timeout = 10.0
    
    async def validate_api_key(self, provider: str, api_key: str) -> APIKeyValidationResult:
        """Validate an API key for a specific provider."""
        if not api_key or not api_key.strip():
            return APIKeyValidationResult(
                is_valid=False,
                error_message="API key cannot be empty",
                provider_name=provider
            )
        
        if provider not in self.validation_endpoints:
            return APIKeyValidationResult(
                is_valid=False,
                error_message=f"Provider '{provider}' is not supported",
                provider_name=provider
            )
        
        try:
            endpoint_config = self.validation_endpoints[provider]
            url = endpoint_config['url']
            headers = endpoint_config['headers_fn'](api_key)
            method = endpoint_config.get('method', 'GET')
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                kwargs = {'headers': headers}
                
                # Add query parameters if needed
                if 'params_fn' in endpoint_config:
                    kwargs['params'] = endpoint_config['params_fn'](api_key)
                
                # Add data for POST requests
                if method == 'POST' and 'data' in endpoint_config:
                    kwargs['json'] = endpoint_config['data']
                
                async with session.request(method, url, **kwargs) as response:
                    # Consider 200-299 as valid, and some specific error codes as invalid API key
                    if response.status < 300:
                        logger.info(f"API key validation successful for {provider}")
                        return APIKeyValidationResult(
                            is_valid=True,
                            provider_name=provider
                        )
                    elif response.status in [401, 403]:
                        logger.warning(f"Invalid API key for {provider}: {response.status}")
                        return APIKeyValidationResult(
                            is_valid=False,
                            error_message="Invalid API key - please check your credentials",
                            provider_name=provider
                        )
                    else:
                        logger.warning(f"Unexpected response for {provider}: {response.status}")
                        # For other errors, we'll assume the key might be valid but service unavailable
                        return APIKeyValidationResult(
                            is_valid=True,  # Assume valid if we can't determine
                            error_message=f"Could not verify API key (service returned {response.status})",
                            provider_name=provider
                        )
                        
        except asyncio.TimeoutError:
            logger.warning(f"Timeout validating API key for {provider}")
            return APIKeyValidationResult(
                is_valid=True,  # Assume valid if timeout
                error_message="Validation timed out - key might be valid",
                provider_name=provider
            )
        except Exception as e:
            logger.error(f"Error validating API key for {provider}: {str(e)}")
            return APIKeyValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}",
                provider_name=provider
            )
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported providers."""
        return list(self.validation_endpoints.keys())
    
    def validate_api_key_sync(self, provider: str, api_key: str) -> APIKeyValidationResult:
        """Synchronous wrapper for API key validation."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.validate_api_key(provider, api_key))
                    )
                    return future.result(timeout=self.timeout + 5)
            else:
                return loop.run_until_complete(self.validate_api_key(provider, api_key))
        except Exception as e:
            logger.error(f"Error in sync validation: {str(e)}")
            return APIKeyValidationResult(
                is_valid=False,
                error_message=f"Validation failed: {str(e)}",
                provider_name=provider
            )
