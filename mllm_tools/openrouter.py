import os
import re
from typing import List, Dict, Any, Optional, Union
import io
import base64
from PIL import Image
import mimetypes
from litellm import completion, completion_cost
from dotenv import load_dotenv

load_dotenv()

class OpenRouterWrapper:
    """
    OpenRouter wrapper using LiteLLM for various language models.
    Compatible with the existing wrapper interface.
    """
    
    def __init__(
        self, 
        model_name: str = "openrouter/deepseek/deepseek-chat-v3-0324:free",
        temperature: float = 0.7,
        print_cost: bool = False,
        verbose: bool = False,
        use_langfuse: bool = True,
        site_url: str = "",
        app_name: str = "Theory2Manim"
    ):
        """
        Initialize OpenRouter wrapper.
        
        Args:
            model_name: OpenRouter model name (with openrouter/ prefix)
            temperature: Temperature for completion
            print_cost: Whether to print the cost of the completion
            verbose: Whether to print verbose output
            use_langfuse: Whether to enable Langfuse logging
            site_url: Optional site URL for tracking
            app_name: Optional app name for tracking
        """
        self.model_name = model_name
        self.temperature = temperature
        self.print_cost = print_cost
        self.verbose = verbose
        self.accumulated_cost = 0
        
        # Setup OpenRouter environment variables
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No OPENROUTER_API_KEY found. Please set the environment variable.")
        
        os.environ["OPENROUTER_API_KEY"] = api_key
        os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"
        
        if site_url or os.getenv("OR_SITE_URL"):
            os.environ["OR_SITE_URL"] = site_url or os.getenv("OR_SITE_URL", "")
        if app_name:
            os.environ["OR_APP_NAME"] = app_name
            
        if self.verbose:
            os.environ['LITELLM_LOG'] = 'DEBUG'
            
        # Set langfuse callback only if enabled
        if use_langfuse:
            import litellm
            litellm.success_callback = ["langfuse"]
            litellm.failure_callback = ["langfuse"]
    
    def _encode_file(self, file_path: Union[str, Image.Image]) -> str:
        """
        Encode local file or PIL Image to base64 string
        
        Args:
            file_path: Path to local file or PIL Image object
            
        Returns:
            Base64 encoded file string
        """
        if isinstance(file_path, Image.Image):
            buffered = io.BytesIO()
            file_path.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")
    
    def _get_mime_type(self, file_path: str) -> str:
        """
        Get the MIME type of a file based on its extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type as a string (e.g., "image/jpeg", "audio/mp3")
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            raise ValueError(f"Unsupported file type: {file_path}")
        return mime_type
    
    def __call__(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process messages and return completion
        
        Args:
            messages: List of message dictionaries with 'type' and 'content' keys
            metadata: Optional metadata to pass to completion
        
        Returns:
            Generated text response
        """
        if metadata is None:
            metadata = {}
        metadata["trace_name"] = f"openrouter-completion-{self.model_name}"
        
        # Convert messages to LiteLLM format
        formatted_messages = []
        for msg in messages:
            if msg["type"] == "text":
                formatted_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": msg["content"]}]
                })
            elif msg["type"] in ["image", "audio", "video"]:
                # Check if content is a local file path or PIL Image
                if isinstance(msg["content"], Image.Image) or os.path.isfile(msg["content"]):
                    try:
                        if isinstance(msg["content"], Image.Image):
                            mime_type = "image/png"
                        else:
                            mime_type = self._get_mime_type(msg["content"])
                        base64_data = self._encode_file(msg["content"])
                        data_url = f"data:{mime_type};base64,{base64_data}"
                    except ValueError as e:
                        print(f"Error processing file {msg['content']}: {e}")
                        continue
                else:
                    data_url = msg["content"]
                
                # Format for vision models
                if msg["type"] == "image":
                    formatted_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                    "detail": "high"
                                }
                            }
                        ]
                    })
                else:
                    # For audio/video, treat as text for now
                    formatted_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": f"[{msg['type'].upper()}]: {msg['content']}"}]
                    })

        try:
            response = completion(
                model=self.model_name,
                messages=formatted_messages,
                temperature=self.temperature,
                metadata=metadata,
                max_retries=99
            )
            if self.print_cost:
                # Calculate and print cost
                cost = completion_cost(completion_response=response)
                self.accumulated_cost += cost
                print(f"Accumulated Cost: ${self.accumulated_cost:.10f}")
            
            content = response.choices[0].message.content
            if content is None:
                print(f"Got null response from model. Full response: {response}")
                return "Error: Received null response from model"
            
            # Check if the response contains error messages about unmapped models
            if "This model isn't mapped yet" in content or "model isn't mapped" in content.lower():
                error_msg = f"Error: Model {self.model_name} is not supported by LiteLLM. Please use a supported model."
                print(error_msg)
                return error_msg
            
            return content
        
        except Exception as e:
            print(f"Error in OpenRouter completion: {e}")
            return f"Error: {str(e)}"


class OpenRouterClient:
    """
    Legacy OpenRouter client for backward compatibility.
    """
    
    def __init__(self, api_key: str, site_url: str = "", app_name: str = "Theory2Manim"):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            site_url: Optional site URL for tracking
            app_name: Optional app name for tracking
        """
        os.environ["OPENROUTER_API_KEY"] = api_key
        os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"
        
        if site_url:
            os.environ["OR_SITE_URL"] = site_url
        if app_name:
            os.environ["OR_APP_NAME"] = app_name
    
    def complete(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "openrouter/openai/gpt-3.5-turbo",
        transforms: Optional[List[str]] = None,
        route: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Generate completion using OpenRouter model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name (with openrouter/ prefix)
            transforms: Optional transforms to apply
            route: Optional route specification
            **kwargs: Additional parameters for completion
            
        Returns:
            Completion response
        """
        params = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        if transforms:
            params["transforms"] = transforms
        if route:
            params["route"] = route
            
        return completion(**params)

# Convenience functions for common models
def ds_r1(messages: List[Dict[str, str]], **kwargs) -> Any:
    """Use GPT-3.5 Turbo via OpenRouter"""
    client = OpenRouterClient(os.environ.get("OPENROUTER_API_KEY", ""))
    return client.complete(messages, "deepseek/deepseek-r1:free", **kwargs)

def ds_v3(messages: List[Dict[str, str]], **kwargs) -> Any:
    """Use GPT-4 via OpenRouter"""
    client = OpenRouterClient(os.environ.get("OPENROUTER_API_KEY", ""))
    return client.complete(messages, "deepseek/deepseek-chat-v3-0324:free", **kwargs)

def qwen3(messages: List[Dict[str, str]], **kwargs) -> Any:
    """Use Claude-2 via OpenRouter"""
    client = OpenRouterClient(os.environ.get("OPENROUTER_API_KEY", ""))
    return client.complete(messages, "qwen/qwen3-235b-a22b:free", **kwargs)


