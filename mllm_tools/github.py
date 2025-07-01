# filepath: d:\Theory2Manim-2\Theory2Manim\mllm_tools\github.py
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

load_dotenv()

class GitHubModelsWrapper:
    """Wrapper for GitHub Models using LiteLLM to support multiple GitHub hosted models"""
    
    def __init__(
        self,
        model_name: str = "github/gpt-4o",
        temperature: float = 0.7,
        print_cost: bool = False,
        verbose: bool = False,
        use_langfuse: bool = True,
        github_token: Optional[str] = None
    ):
        """
        Initialize the GitHub Models wrapper
        
        Args:
            model_name: Name of the GitHub model to use (e.g. "github/gpt-4o", "github/gpt-4o-mini", 
                       "github/o1-preview", "github/claude-3-5-sonnet", "github/phi-3.5-mini-instruct")
            temperature: Temperature for completion
            print_cost: Whether to print the cost of the completion
            verbose: Whether to print verbose output
            use_langfuse: Whether to enable Langfuse logging
            github_token: GitHub token for authentication (if not provided, will use GITHUB_TOKEN env var)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.print_cost = print_cost
        self.verbose = verbose
        self.accumulated_cost = 0
        
        # Set up GitHub token
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        if not self.github_token:
            raise ValueError("GitHub token is required. Please set GITHUB_TOKEN environment variable or pass github_token parameter.")
        
        # Set environment variable for LiteLLM
        os.environ['GITHUB_TOKEN'] = self.github_token

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
            MIME type as a string (e.g., "image/jpeg", "audio/mp3")
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
            "claude-3-5-sonnet",
            "claude-3-haiku"
        ]
        
        # Extract model name without the github/ prefix
        clean_model_name = model_name.replace("github/", "")
        return any(vision_model in clean_model_name for vision_model in vision_models)

    def __call__(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process messages and return completion
        
        Args:
            messages: List of message dictionaries with 'type' and 'content' keys
            metadata: Optional metadata to pass to litellm completion, e.g. for Langfuse tracking
        
        Returns:
            Generated text response
        """
        if metadata is None:
            metadata = {}
        metadata["trace_name"] = f"github-models-completion-{self.model_name}"
        
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
            else:
                raise ValueError(f"Unsupported message type: {msg['type']}. GitHub models currently support 'text' and 'image' types.")

        try:
            # Check if it's an o-series model (like o1-preview, o1-mini)
            if (re.match(r".*o1.*", self.model_name)):
                # O-series models don't support temperature and have reasoning_effort
                response = completion(
                    model=self.model_name,
                    messages=formatted_messages,
                    reasoning_effort="medium",  # Options: "low", "medium", "high"
                    metadata=metadata,
                    max_retries=3
                )
            else:
                response = completion(
                    model=self.model_name,
                    messages=formatted_messages,
                    temperature=self.temperature,
                    metadata=metadata,
                    max_retries=3
                )
                
            if self.print_cost:
                try:
                    # Note: GitHub Models may not provide cost information
                    cost = completion_cost(completion_response=response)
                    if cost is not None:
                        self.accumulated_cost += cost
                        print(f"Cost: ${float(cost):.10f}")
                        print(f"Accumulated Cost: ${self.accumulated_cost:.10f}")
                    else:
                        print("Cost information not available for GitHub Models")
                except Exception as e:
                    print(f"Could not calculate cost: {e}")
                
            content = response.choices[0].message.content
            if content is None:
                print(f"Got null response from GitHub model. Full response: {response}")
                return ""
            return content
        
        except Exception as e:
            print(f"Error in GitHub model completion: {e}")
            return str(e)

def create_github_model_wrapper(model_name: str = "github/gpt-4o", **kwargs) -> GitHubModelsWrapper:
    """
    Convenience function to create a GitHub Models wrapper
    
    Args:
        model_name: GitHub model name (e.g., "github/gpt-4o", "github/claude-3-5-sonnet")
        **kwargs: Additional arguments passed to GitHubModelsWrapper
        
    Returns:
        Configured GitHubModelsWrapper instance
        
    Example:
        >>> # Create a wrapper for GPT-4o
        >>> wrapper = create_github_model_wrapper("github/gpt-4o", temperature=0.3)
        >>> 
        >>> # Use it for text generation
        >>> response = wrapper([{"type": "text", "content": "Explain quantum computing"}])
        >>> 
        >>> # Use it for vision (if model supports it)
        >>> response = wrapper([
        ...     {"type": "text", "content": "What's in this image?"},
        ...     {"type": "image", "content": "path/to/image.jpg"}
        ... ])
    """
    return GitHubModelsWrapper(model_name=model_name, **kwargs)

# Available GitHub Models (as of the documentation)
AVAILABLE_MODELS = {
    # GPT Models
    "gpt-4o": "github/gpt-4o",
    "gpt-4o-mini": "github/gpt-4o-mini",
    "o1-preview": "github/o1-preview", 
    "o1-mini": "github/o1-mini",
    "gpt-4.1": "github/gpt-4.1",
    
    
    # Phi Models
    "phi-3-5-mini-instruct": "github/phi-3.5-mini-instruct",
    "phi-3-5-moe-instruct": "github/phi-3.5-moe-instruct",
    
    # Llama Models  
    "llama-3.1-405b-instruct": "github/llama-3.1-405b-instruct",
    "llama-3.1-70b-instruct": "github/llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct": "github/llama-3.1-8b-instruct",
    
    # Mistral Models
    "mistral-large": "github/mistral-large",
    "mistral-large-2407": "github/mistral-large-2407",
    "mistral-nemo": "github/mistral-nemo",
    "mistral-small": "github/mistral-small",
    
    # Cohere Models
    "cohere-command-r": "github/cohere-command-r",
    "cohere-command-r-plus": "github/cohere-command-r-plus",
    
    # AI21 Models
    "ai21-jamba-1.5-large": "github/ai21-jamba-1.5-large",
    "ai21-jamba-1.5-mini": "github/ai21-jamba-1.5-mini"
}

def list_available_models() -> Dict[str, str]:
    """
    Get a dictionary of available GitHub models
    
    Returns:
        Dictionary mapping friendly names to full model names
    """
    return AVAILABLE_MODELS.copy()

if __name__ == "__main__":
    # Example usage
    print("Available GitHub Models:")
    for friendly_name, full_name in AVAILABLE_MODELS.items():
        print(f"  {friendly_name}: {full_name}")
    
    # Example of creating a wrapper (requires GITHUB_TOKEN environment variable)
    try:
        wrapper = create_github_model_wrapper("github/gpt-4o-mini", temperature=0.3)
        print("\nGitHub Models wrapper created successfully!")
        
        # Test with a simple text prompt
        response = wrapper([{"type": "text", "content": "Hello! Can you confirm you're working?"}])
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error creating wrapper: {e}")
        print("Make sure to set GITHUB_TOKEN environment variable")