# filepath: d:\Theory2Manim-2\Theory2Manim\mllm_tools\openai.py
import json
import re
from typing import List, Dict, Any, Union, Optional
import io
import os
import base64
from PIL import Image
import mimetypes
import litellm
from litellm import completion, completion_cost
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Note: Environment variables should be loaded from .env file or set manually using os.environ

class OpenAIWrapper:
    """Wrapper for OpenAI using LiteLLM to support all OpenAI models with unified interface"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        print_cost: bool = False,
        verbose: bool = False,
        use_langfuse: bool = True,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        use_github_token: bool = True,
        github_token: Optional[str] = os.getenv('GITHUB_TOKEN')
    ):
        """
        Initialize the OpenAI wrapper
        
        Args:
            model_name: Name of the OpenAI model to use (e.g. "gpt-4o", "gpt-4o-mini", 
                       "gpt-3.5-turbo", "o1-preview", "o1-mini", "dall-e-3")
            temperature: Temperature for completion (ignored for o1 models)
            print_cost: Whether to print the cost of the completion
            verbose: Whether to print verbose output
            use_langfuse: Whether to enable Langfuse logging
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
            organization: OpenAI organization ID (optional)
            base_url: Custom base URL for OpenAI API (optional, for proxies)
            use_github_token: Whether to use GitHub AI model inference endpoint
            github_token: GitHub token (if not provided, will use GITHUB_TOKEN env var)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.print_cost = print_cost
        self.verbose = verbose
        self.accumulated_cost = 0
        self.use_github_token = use_github_token
        
        # Configure API based on whether using GitHub token or OpenAI API
        if use_github_token:
            # Set up GitHub token and endpoint
            self.github_token = github_token or os.getenv('GITHUB_TOKEN')
            if not self.github_token:
                raise ValueError("GitHub token is required when use_github_token=True. Please set GITHUB_TOKEN environment variable or pass github_token parameter.")
            
            # Set GitHub AI inference endpoint
            self.base_url = "https://models.github.ai/inference"
            self.api_key = self.github_token
            
            # Set environment variables for LiteLLM to use GitHub endpoint
            os.environ['OPENAI_API_KEY'] = self.github_token
            os.environ['OPENAI_BASE_URL'] = self.base_url
            
            # Adjust model name for GitHub endpoint (add openai/ prefix if not present)
            if not self.model_name.startswith("openai/"):
                self.model_name = f"openai/{self.model_name}"
                
        else:
            # Original OpenAI API setup
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
            # Set environment variables for LiteLLM
            os.environ['OPENAI_API_KEY'] = self.api_key
            
            # Set optional custom base URL
            if base_url:
                os.environ['OPENAI_BASE_URL'] = base_url
                self.base_url = base_url
            else:
                self.base_url = os.getenv('OPENAI_BASE_URL')
        
        # Set optional organization (only for OpenAI, not GitHub)
        if not use_github_token:
            if organization:
                os.environ['OPENAI_ORGANIZATION'] = organization
                self.organization = organization
            else:
                self.organization = os.getenv('OPENAI_ORGANIZATION')
        else:
            self.organization = None

        if self.verbose:
            os.environ['LITELLM_LOG'] = 'DEBUG'
        
        # Set langfuse callback only if enabled
        if use_langfuse:
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
            MIME type as a string (e.g., "image/jpeg", "application/pdf")
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            raise ValueError(f"Unsupported file type: {file_path}")
        return mime_type

    def _supports_vision(self, model_name: str) -> bool:
        """
        Check if the model supports vision/image processing
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model supports vision, False otherwise
        """
        vision_models = [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-vision-preview",
            "gpt-4-turbo",
            "gpt-4-turbo-vision"
        ]
        
        return any(vision_model in model_name for vision_model in vision_models)

    def _supports_files(self, model_name: str) -> bool:
        """
        Check if the model supports file processing (PDFs, documents)
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model supports file processing, False otherwise
        """
        file_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo"
        ]
        
        return any(file_model in model_name for file_model in file_models)

    def _is_o1_model(self, model_name: str) -> bool:
        """
        Check if the model is an o1 series model (reasoning models)
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if it's an o1 model, False otherwise
        """
        return "o1" in model_name

    def __call__(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """
        Process messages and return completion
        
        Args:
            messages: List of message dictionaries with 'type' and 'content' keys
            metadata: Optional metadata to pass to litellm completion, e.g. for Langfuse tracking
            **kwargs: Additional parameters for completion (max_tokens, stream, etc.)
        
        Returns:
            Generated text response
        """
        if metadata is None:
            metadata = {}
        metadata["trace_name"] = f"openai-completion-{self.model_name}"
        
        # Convert messages to LiteLLM format
        formatted_messages = []
        
        for msg in messages:
            if msg["type"] == "text":
                formatted_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": msg["content"]}]
                })
            elif msg["type"] == "image":
                # Check if model supports vision
                if not self._supports_vision(self.model_name):
                    raise ValueError(f"Model {self.model_name} does not support image processing")
                
                # Check if content is a local file path or PIL Image
                if isinstance(msg["content"], Image.Image) or (isinstance(msg["content"], str) and os.path.isfile(msg["content"])):
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
                    # Assume it's already a URL or base64 string
                    data_url = msg["content"]

                # Format for vision-capable models
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
            elif msg["type"] == "file":
                # Check if model supports file processing
                if not self._supports_files(self.model_name):
                    raise ValueError(f"Model {self.model_name} does not support file processing")
                
                # Handle file content (PDF, documents, etc.)
                if os.path.isfile(msg["content"]):
                    try:
                        mime_type = self._get_mime_type(msg["content"])
                        base64_data = self._encode_file(msg["content"])
                        
                        # Use the file format for document processing
                        formatted_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "file",
                                    "file": {
                                        "filename": os.path.basename(msg["content"]),
                                        "file_data": f"data:{mime_type};base64,{base64_data}",
                                    }
                                }
                            ]
                        })
                    except ValueError as e:
                        print(f"Error processing file {msg['content']}: {e}")
                        continue
                else:
                    raise ValueError(f"File not found: {msg['content']}")
            else:
                raise ValueError(f"Unsupported message type: {msg['type']}. OpenAI models support 'text', 'image', and 'file' types.")

        try:
            # Prepare completion parameters
            completion_params = {
                "model": self.model_name,
                "messages": formatted_messages,
                "metadata": metadata,
                "max_retries": 3
            }
            
            # Add additional kwargs
            completion_params.update(kwargs)
            
            # Check if it's an o1 series model (reasoning models)
            if self._is_o1_model(self.model_name):
                # O1 models don't support temperature and have reasoning_effort
                if "reasoning_effort" not in completion_params:
                    completion_params["reasoning_effort"] = "medium"  # Options: "low", "medium", "high"
                # Remove temperature if it was added via kwargs
                completion_params.pop("temperature", None)
            else:
                # Regular models support temperature
                if "temperature" not in completion_params:
                    completion_params["temperature"] = self.temperature
                    
            response = completion(**completion_params)
                
            if self.print_cost:
                try:
                    cost = completion_cost(completion_response=response)
                    if cost is not None:
                        self.accumulated_cost += cost
                        print(f"Cost: ${float(cost):.10f}")
                        print(f"Accumulated Cost: ${self.accumulated_cost:.10f}")
                    else:
                        print("Cost information not available")
                except Exception as e:
                    print(f"Could not calculate cost: {e}")
                
            content = response.choices[0].message.content
            if content is None:
                print(f"Got null response from OpenAI model. Full response: {response}")
                return ""
            return content
        
        except Exception as e:
            print(f"Error in OpenAI model completion: {e}")
            return str(e)

    def stream_completion(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Process messages and return streaming completion
        
        Args:
            messages: List of message dictionaries with 'type' and 'content' keys
            metadata: Optional metadata to pass to litellm completion
            **kwargs: Additional parameters for completion
        
        Yields:
            Streaming response chunks
        """
        kwargs["stream"] = True
        
        # Use the same message formatting as regular completion
        if metadata is None:
            metadata = {}
        metadata["trace_name"] = f"openai-streaming-{self.model_name}"
        
        try:
            # Convert messages to the same format as __call__
            formatted_messages = []
            
            for msg in messages:
                if msg["type"] == "text":
                    formatted_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                elif msg["type"] == "image":
                    if not self._supports_vision(self.model_name):
                        raise ValueError(f"Model {self.model_name} does not support image processing")
                    
                    if isinstance(msg["content"], Image.Image) or (isinstance(msg["content"], str) and os.path.isfile(msg["content"])):
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

            # Prepare completion parameters
            completion_params = {
                "model": self.model_name,
                "messages": formatted_messages,
                "metadata": metadata,
                "max_retries": 3,
                "stream": True
            }
            
            # Add additional kwargs
            completion_params.update(kwargs)
            
            # Handle o1 models
            if self._is_o1_model(self.model_name):
                if "reasoning_effort" not in completion_params:
                    completion_params["reasoning_effort"] = "medium"
                completion_params.pop("temperature", None)
            else:
                if "temperature" not in completion_params:
                    completion_params["temperature"] = self.temperature
                    
            response = completion(**completion_params)
            
            # Yield streaming chunks
            for chunk in response:
                yield chunk
                
        except Exception as e:
            print(f"Error in OpenAI streaming completion: {e}")
            yield {"error": str(e)}

def create_openai_wrapper(model_name: str = "gpt-4o", use_github: bool = False, **kwargs) -> OpenAIWrapper:
    """
    Convenience function to create an OpenAI wrapper
    
    Args:
        model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo")
        use_github: Whether to use GitHub's AI model inference endpoint
        **kwargs: Additional arguments passed to OpenAIWrapper
        
    Returns:
        Configured OpenAIWrapper instance
        
    Example:
        >>> # Create a wrapper for GPT-4o using regular OpenAI
        >>> wrapper = create_openai_wrapper("gpt-4o", temperature=0.3)
        >>> 
        >>> # Create a wrapper for GPT-4o using GitHub AI models
        >>> wrapper = create_openai_wrapper("gpt-4o", use_github=True, temperature=0.3)
        >>> 
        >>> # Use it for text generation
        >>> response = wrapper([{"type": "text", "content": "Explain quantum computing"}])
        >>> 
        >>> # Use it for vision (if model supports it)
        >>> response = wrapper([
        ...     {"type": "text", "content": "What's in this image?"},
        ...     {"type": "image", "content": "path/to/image.jpg"}
        ... ])
        >>> 
        >>> # Use it for file processing (PDFs, etc.)
        >>> response = wrapper([
        ...     {"type": "text", "content": "Summarize this document"},
        ...     {"type": "file", "content": "path/to/document.pdf"}
        ... ])
    """
    return OpenAIWrapper(model_name=model_name, use_github_token=use_github, **kwargs)

# Available OpenAI Models
AVAILABLE_MODELS = {
    # GPT-4 Models
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4": "gpt-4",
    "gpt-4-vision-preview": "gpt-4-vision-preview",
    
    # O1 Reasoning Models
    "o1-preview": "o1-preview",
    "o1-mini": "o1-mini",
    
    # GPT-3.5 Models
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-3.5-turbo-instruct": "gpt-3.5-turbo-instruct",
    
    # Image Generation Models
    "dall-e-3": "dall-e-3",
    "dall-e-2": "dall-e-2",
    
    # Embedding Models
    "text-embedding-3-large": "text-embedding-3-large",
    "text-embedding-3-small": "text-embedding-3-small", 
    "text-embedding-ada-002": "text-embedding-ada-002",
    
    # Audio Models
    "whisper-1": "whisper-1",
    "tts-1": "tts-1",
    "tts-1-hd": "tts-1-hd",
}

def create_github_openai_wrapper(model_name: str = "gpt-4o", **kwargs) -> OpenAIWrapper:
    """
    Convenience function to create an OpenAI wrapper using GitHub's AI model inference
    
    Args:
        model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo")
        **kwargs: Additional arguments passed to OpenAIWrapper
        
    Returns:
        Configured OpenAIWrapper instance using GitHub endpoint
        
    Example:
        >>> # Create a wrapper for GPT-4o using GitHub AI models
        >>> wrapper = create_github_openai_wrapper("gpt-4o", temperature=0.3)
        >>> 
        >>> # Use it for text generation
        >>> response = wrapper([{"type": "text", "content": "What is the capital of France?"}])
    """
    return OpenAIWrapper(model_name=model_name, use_github_token=True, **kwargs)

def list_available_models() -> Dict[str, str]:
    """
    Get a dictionary of available OpenAI models
    
    Returns:
        Dictionary mapping model names to their identifiers
    """
    return AVAILABLE_MODELS.copy()

def get_model_capabilities(model_name: str) -> Dict[str, bool]:
    """
    Get the capabilities of a specific model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of capabilities (vision, files, reasoning, etc.)
    """
    wrapper = OpenAIWrapper(model_name=model_name)
    
    return {
        "vision": wrapper._supports_vision(model_name),
        "files": wrapper._supports_files(model_name),
        "reasoning": wrapper._is_o1_model(model_name),
        "streaming": not wrapper._is_o1_model(model_name),  # O1 models don't support streaming
        "temperature": not wrapper._is_o1_model(model_name),  # O1 models don't support temperature
    }

if __name__ == "__main__":
    # Example usage
    print("Available OpenAI Models:")
    for model_name, model_id in AVAILABLE_MODELS.items():
        capabilities = get_model_capabilities(model_name)
        print(f"  {model_name} ({model_id}): {capabilities}")
    
    print("\n" + "="*50)
    print("Testing OpenAI wrapper...")
    
    # Example 1: Regular OpenAI (requires OPENAI_API_KEY environment variable)
    try:
        print("\n1. Testing regular OpenAI wrapper:")
        wrapper = create_openai_wrapper("gpt-4o-mini", temperature=0.3)
        print("Regular OpenAI wrapper created successfully!")
        
        # Test with a simple text prompt
        response = wrapper([{"type": "text", "content": "Hello! Can you confirm you're working?"}])
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error creating regular OpenAI wrapper: {e}")
        print("Make sure to set OPENAI_API_KEY environment variable")
    
    # Example 2: GitHub AI models (requires GITHUB_TOKEN environment variable)
    try:
        print("\n2. Testing GitHub AI models wrapper:")
        github_wrapper = create_github_openai_wrapper("gpt-4o", temperature=1.0)
        print("GitHub OpenAI wrapper created successfully!")
        
        # Test with a simple text prompt
        response = github_wrapper([{
            "type": "text", 
            "content": "What is the capital of France?"
        }])
        print(f"GitHub Response: {response}")
        
    except Exception as e:
        print(f"Error creating GitHub wrapper: {e}")
        print("Make sure to set GITHUB_TOKEN environment variable")
      # Example 3: Manual GitHub configuration
    try:
        print("\n3. Testing manual GitHub configuration:")
        manual_wrapper = OpenAIWrapper(
            model_name="openai/gpt-4o",
            use_github_token=True,
            temperature=1.0,
            verbose=False
        )
        print("Manual GitHub wrapper created successfully!")
        
    except Exception as e:
        print(f"Error creating manual GitHub wrapper: {e}")
        print("Make sure to set GITHUB_TOKEN environment variable")