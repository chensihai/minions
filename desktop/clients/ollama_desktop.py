import os
import json
import requests
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

# Import the Usage class
from minions.usage import Usage

class DesktopOllamaClient:
    """Desktop-specific Ollama client that communicates directly with the Ollama API via HTTP."""
    
    def __init__(
        self,
        model_name: str = "llama-3.2",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        num_ctx: int = 4096,
        structured_output_schema: Optional[Any] = None,
        use_async: bool = False,
    ):
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Get configuration from environment variables
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_DEFAULT_MODEL", "deepseek-r1:1.5b")
        
        # Store other parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.structured_output_schema = structured_output_schema
        
        # Create a session for connection reuse
        self.session = requests.Session()
        
        # Check if Ollama is running
        self._check_ollama_running()
        
        # Ensure model is available
        self._ensure_model_available()
        
        # Print configuration for debugging
        print(f"Desktop Ollama Client initialized with:")
        print(f"  Base URL: {self.base_url}")
        print(f"  Model: {self.model_name}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max Tokens: {self.max_tokens}")
    
    def _check_ollama_running(self):
        """Check if Ollama is running and accessible."""
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=5)
            response.raise_for_status()
            version = response.json().get("version", "unknown")
            print(f"Successfully connected to Ollama v{version} at {self.base_url}")
            return True
        except requests.RequestException as e:
            error_message = (
                f"Failed to connect to Ollama at {self.base_url}.\n"
                "Please ensure Ollama is running in WSL or locally:\n"
                "1. Check that the Ollama service is running\n"
                "2. Verify the URL in your .env file (OLLAMA_BASE_URL)\n"
                f"Error details: {str(e)}"
            )
            raise ConnectionError(error_message) from e
    
    def _ensure_model_available(self):
        """Check if the model is available, pull if necessary."""
        try:
            # Try to get model info
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                model_info = response.json()
                print(f"Model {self.model_name} is available (size: {model_info.get('size', 'unknown')})")
                return True
            
        except requests.RequestException:
            # Model not found, need to pull it
            pass
        
        # If we get here, we need to pull the model
        print(f"Model {self.model_name} not found. Pulling (this may take a while)...")
        try:
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                stream=True,
                timeout=None  # No timeout for model pulling
            )
            response.raise_for_status()
            
            # Process the streaming response to show progress
            for line in response.iter_lines():
                if line:
                    progress = json.loads(line.decode('utf-8'))
                    if 'status' in progress:
                        print(f"Pull status: {progress['status']}")
                    if 'completed' in progress and 'total' in progress:
                        percent = (progress['completed'] / progress['total']) * 100
                        print(f"Downloaded: {percent:.1f}%")
            
            print(f"Successfully pulled model {self.model_name}")
            return True
            
        except requests.RequestException as e:
            error_message = f"Failed to pull model {self.model_name}: {str(e)}"
            raise RuntimeError(error_message) from e
    
    def chat(self, messages, temperature=None, max_tokens=None, stream=False, response_format=None, **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """Send a chat request to the Ollama API."""
        if temperature is None:
            temperature = self.temperature
        
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        # If we're sending multiple messages at once (as in worker_chats), process them individually
        if isinstance(messages, list) and all(isinstance(msg, dict) for msg in messages):
            # This is a list of individual messages (like worker_chats)
            responses = []
            usage_total = Usage(prompt_tokens=0, completion_tokens=0)
            done_reasons = []
            
            for message in messages:
                # Process each message individually
                single_message_list = [message]
                
                # Call chat with a single message
                content_list, usage, reason = self._process_single_chat(
                    single_message_list, 
                    temperature, 
                    max_tokens, 
                    stream, 
                    response_format, 
                    **kwargs
                )
                
                # Format the response as JSON if needed
                if content_list and content_list[0]:
                    # Format the response as a JSON string for JobOutput
                    content = self._format_as_job_output_json(content_list[0])
                    responses.append(content)
                else:
                    responses.append("")
                
                # Accumulate usage
                usage_total.prompt_tokens += usage.prompt_tokens
                usage_total.completion_tokens += usage.completion_tokens
                
                # Add done reason
                done_reasons.append(reason[0] if reason else "stop")
            
            return responses, usage_total, done_reasons
        else:
            # This is a single message or a list of messages in a conversation
            return self._process_single_chat(messages, temperature, max_tokens, stream, response_format, **kwargs)
    
    def _process_single_chat(self, messages, temperature=None, max_tokens=None, stream=False, response_format=None, **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """Process a single chat request."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream
        }
        
        # Add any additional kwargs to the payload
        for key, value in kwargs.items():
            payload[key] = value
            
        # Handle response_format if provided
        # Note: Ollama doesn't natively support response_format like OpenAI,
        # but we can add a system message to request JSON output
        if response_format and response_format.get("type") == "json_object":
            # Add a system message requesting JSON output
            system_msg = {"role": "system", "content": "Please provide your response in valid JSON format."}
            if any(msg.get("role") == "system" for msg in messages):
                # Update existing system message
                for msg in messages:
                    if msg.get("role") == "system":
                        msg["content"] += " Please provide your response in valid JSON format."
                        break
            else:
                # Add new system message
                payload["messages"] = [system_msg] + messages
        
        try:
            if stream:
                response = self.session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=None
                )
                response.raise_for_status()
                return self._process_streaming_response(response)
            else:
                response = self.session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=None
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract content and create usage object
                content = result['message']['content']
                
                # Create a Usage object with token counts
                # Ollama API returns prompt_eval_count and eval_count
                usage = Usage(
                    prompt_tokens=result.get('prompt_eval_count', 0),
                    completion_tokens=result.get('eval_count', 0)
                )
                
                # Return tuple of responses, usage, and done_reasons
                return [content], usage, [result.get('done_reason', 'stop')]
                
        except requests.RequestException as e:
            error_message = f"Error communicating with Ollama: {str(e)}"
            self.logger.error(error_message)
            raise RuntimeError(error_message) from e
    
    def _process_streaming_response(self, response):
        """Process a streaming response from the Ollama API."""
        full_response = {"message": {"content": ""}}
        prompt_tokens = 0
        completion_tokens = 0
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                if 'message' in chunk and 'content' in chunk['message']:
                    full_response["message"]["content"] += chunk["message"]["content"]
                    # Track token counts if available
                    if 'prompt_eval_count' in chunk:
                        prompt_tokens = chunk['prompt_eval_count']
                    if 'eval_count' in chunk:
                        completion_tokens = chunk['eval_count']
                    yield chunk
        
        content = full_response["message"]["content"]
        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        
        # Return tuple of responses, usage, and done_reasons to match the original OllamaClient
        return [content], usage, ["stop"]
    
    def _format_as_job_output_json(self, content):
        """
        Format a text response as a valid JSON string for JobOutput.
        Enhanced to handle various formats including markdown and code blocks.
        """
        import re
        
        # Handle empty content
        if not content or content.isspace():
            # Return a default JobOutput
            return json.dumps({
                "answer": "No response received",
                "explanation": "The model returned an empty response",
                "citation": None
            })
        
        # First try to find JSON in code blocks
        json_pattern = re.compile(r'```(?:json)?\s*({.*?})\s*```', re.DOTALL)
        match = json_pattern.search(content)
        if match:
            try:
                # Validate the extracted JSON
                json_str = match.group(1).strip()
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        
        # Then try parsing the entire content as JSON
        # try:
        #     json.loads(content)
        #     return content
        # except json.JSONDecodeError:
        #     pass
        
        # Look for markdown-style answer and explanation
        answer = None
        explanation = None
        
        # Try markdown bold format: **Answer:** text
        bold_answer = re.search(r'\*\*Answer:\*\*\s*(.*?)(?=\n|$|\*\*)', content, re.IGNORECASE | re.DOTALL)
        if bold_answer:
            answer = bold_answer.group(1).strip()
        
        bold_explanation = re.search(r'\*\*Explanation:\*\*\s*(.*?)(?=\n|$|\*\*)', content, re.IGNORECASE | re.DOTALL)
        if bold_explanation:
            explanation = bold_explanation.group(1).strip()
        
        # Try regular format: Answer: text
        if not answer:
            regular_answer = re.search(r'Answer:\s*(.*?)(?=\n|$)', content, re.IGNORECASE | re.DOTALL)
            if regular_answer:
                answer = regular_answer.group(1).strip()
        
        if not explanation:
            regular_explanation = re.search(r'Explanation:\s*(.*?)(?=\n|$)', content, re.IGNORECASE | re.DOTALL)
            if regular_explanation:
                explanation = regular_explanation.group(1).strip()
        
        # If we still don't have an answer, look for any sentence that might be an answer
        if not answer:
            # Look for sentences containing keywords like "contains", "is", "are", "total"
            answer_candidates = re.findall(r'(?:The\s+)?(?:answer|result|text\s+contains|there\s+are|total)[^.!?]*[.!?]', content, re.IGNORECASE)
            if answer_candidates:
                answer = answer_candidates[0].strip()
        
        # If still no answer, use the first non-empty line that's not part of a code block or think block
        if not answer:
            lines = [line.strip() for line in content.split('\n') 
                    if line.strip() and not line.strip().startswith('```') 
                    and not line.strip().startswith('<think>')]
            if lines:
                # Skip lines that are likely part of a think block or code block
                for line in lines:
                    if not line.startswith(('<', '```', '#', '-', '*')) and len(line) > 10:
                        answer = line
                        break
                if not answer and lines:
                    answer = lines[0]
        
        # If no explanation found, provide a generic one
        if not explanation:
            explanation = "Generated response from model"
        
        # If still no answer found, use a fallback
        if not answer:
            answer = "Unable to extract a clear answer from the response"
        
        # Create the JobOutput JSON
        job_output = {
            "answer": answer,
            "explanation": explanation,
            "citation": None
        }
        
        return json.dumps(job_output)
    
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
