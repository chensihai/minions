import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import openai

# Import the Usage class
from minions.usage import Usage

class DesktopOpenAIClient:
    """Desktop-specific OpenAI client that handles parameter differences."""
    
    def __init__(
        self,
        model: str = "gpt-4o",  # Accept 'model' instead of 'model_name'
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        num_ctx: int = 8192,  # Additional parameter not used but accepted
        structured_output_schema: Optional[Any] = None,  # Additional parameter not used but accepted
        use_async: bool = False,  # Additional parameter not used but accepted
    ):
        """Initialize the Desktop OpenAI client."""
        self.logger = logging.getLogger(__name__)
        
        # Store parameters
        self.model_name = model  # Map 'model' to 'model_name'
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Please provide it as an argument or set the OPENAI_API_KEY environment variable."
            )
        
        # Initialize the OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        
        print(f"Desktop OpenAI Client initialized with model: {self.model_name}")
    
    def chat(self, messages, temperature=None, max_tokens=None, response_format=None, **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """Send a chat request to the OpenAI API."""
        if temperature is None:
            temperature = self.temperature
        
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        try:
            # Build request parameters
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add response_format if provided
            if response_format:
                params["response_format"] = response_format
                
            # Add any additional kwargs
            params.update(kwargs)
            
            response = self.client.chat.completions.create(**params)
            
            # Extract content
            content = response.choices[0].message.content
            
            # Create a Usage object with token counts
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens
            )
            
            # Return tuple of responses, usage, and done_reasons to match the original OpenAIClient
            return [content], usage, ["stop"]
            
        except Exception as e:
            error_message = f"Error communicating with OpenAI: {str(e)}"
            self.logger.error(error_message)
            raise RuntimeError(error_message) from e
    
    def generate(self, prompt, **kwargs) -> Tuple[str, Usage]:
        """Generate a response to a prompt."""
        messages = [{"role": "user", "content": prompt}]
        responses, usage, _ = self.chat(messages, **kwargs)
        
        # Return the first response and the usage
        return responses[0], usage
    
    def run(self, task, context=None, **kwargs) -> Tuple[str, Usage]:
        """Run method to maintain compatibility with the original client."""
        prompt = task
        if context:
            prompt = f"{context}\n\n{task}"
        
        return self.generate(prompt, **kwargs)
