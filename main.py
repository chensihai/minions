import platform
import os
import sys

# Disable MLX on non-macOS platforms
if platform.system() != "Darwin":
    # Create a dummy mlx_lm module to prevent import errors
    class DummyModule:
        def __getattr__(self, name):
            return None
    
    sys.modules['mlx_lm'] = DummyModule()
    os.environ['DISABLE_MLX'] = '1'
    print("MLX LM disabled (not supported on this platform)")

import gi
import threading
import json
import ctypes
from dotenv import load_dotenv
load_dotenv(override=True)  # Force reload environment variables from .env file

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

# Verify API keys are loaded
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Print verification (remove in production)
print(f"OpenAI API Key loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
print(f"Anthropic API Key loaded: {'Yes' if ANTHROPIC_API_KEY else 'No'}")
print(f"Together API Key loaded: {'Yes' if TOGETHER_API_KEY else 'No'}")
print(f"Perplexity API Key loaded: {'Yes' if PERPLEXITY_API_KEY else 'No'}")
print(f"OpenRouter API Key loaded: {'Yes' if OPENROUTER_API_KEY else 'No'}")
print(f"Azure OpenAI API Key loaded: {'Yes' if AZURE_OPENAI_API_KEY else 'No'}")
print(f"Groq API Key loaded: {'Yes' if GROQ_API_KEY else 'No'}")
print(f"DeepSeek API Key loaded: {'Yes' if DEEPSEEK_API_KEY else 'No'}")

# Windows-specific configuration
if os.name == 'nt':
    # Add GTK runtime to PATH
    # os.environ["PATH"] = r"C:\tools\msys64\mingw64\bin;" + os.environ.get("PATH", "")
    
    # Force basic theme and backend
    os.environ["GTK_THEME"] = "win32"
    os.environ["GDK_BACKEND"] = "win32"
    
    # High DPI fix
    ctypes.windll.shcore.SetProcessDpiAwareness(1)

# Debug environment
os.environ["G_MESSAGES_DEBUG"] = "all"
os.environ["GTK_DEBUG"] = "interactive"

# Import Minions related modules
from minions.minion import Minion
from minions.minions import Minions
from minions.minions_mcp import SyncMinionsMCP, MCPConfigManager

# Import client modules
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.clients.anthropic import AnthropicClient
from minions.clients.perplexity import PerplexityAIClient
from minions.clients.openrouter import OpenRouterClient
from minions.clients.together import TogetherClient
from minions.clients.azure_openai import AzureOpenAIClient
from minions.clients.groq import GroqClient
from minions.clients.deepseek import DeepSeekClient

# Additional imports
import time
from openai import OpenAI
from PIL import Image
import io
from pydantic import BaseModel
from typing import List, Optional, Dict

# Import additional libraries for document processing
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import io

# Document processing functions
def extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None


def extract_text_from_image(image_bytes, parent_window=None):
    """Extract text from an image file using pytesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
        import io
        import os
        
        # Set the path to the Tesseract executable
        # First check if it's defined in environment variable
        tesseract_path = os.getenv('TESSERACT_PATH')
        if not tesseract_path:
            # Use the default installation path if not specified in environment
            tesseract_path = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
        
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # For debugging
        print(f"Using Tesseract from: {pytesseract.pytesseract.tesseract_cmd}")
        
        # Check if the file exists
        if not os.path.isfile(pytesseract.pytesseract.tesseract_cmd):
            print(f"WARNING: Tesseract executable not found at {pytesseract.pytesseract.tesseract_cmd}")
            print("Please install Tesseract OCR or set the correct path in TESSERACT_PATH environment variable")
            
            if parent_window:
                # Show a dialog to help the user
                dialog = Gtk.MessageDialog(
                    transient_for=parent_window,
                    modal=True,
                    message_type=Gtk.MessageType.ERROR,
                    buttons=Gtk.ButtonsType.OK,
                    text="Tesseract OCR Not Found"
                )
                dialog.format_secondary_text(
                    f"Tesseract executable not found at {pytesseract.pytesseract.tesseract_cmd}\n\n"
                    "Please install Tesseract OCR from:\n"
                    "https://github.com/UB-Mannheim/tesseract/releases\n\n"
                    "Or set the correct path in your .env file with:\n"
                    "TESSERACT_PATH=path/to/tesseract.exe"
                )
                dialog.run()
                dialog.destroy()
                return "Error: Tesseract OCR not found. Please install it first."
        
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        if parent_window:
            # Show error dialog
            dialog = Gtk.MessageDialog(
                transient_for=parent_window,
                modal=True,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text="OCR Error"
            )
            dialog.format_secondary_text(f"Error processing image: {str(e)}")
            dialog.run()
            dialog.destroy()
        return None


MODEL_MAP = {
    "OpenAI": ["text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"],
    "Anthropic": ["anthropic-cassius-001", "anthropic-cassius-002"],
    "Together": ["together-gpt-001", "together-gpt-002"],
    "Perplexity": ["perplexity-gpt-001", "perplexity-gpt-002"],
    "OpenRouter": ["openrouter-gpt-001", "openrouter-gpt-002"],
    "AzureOpenAI": ["text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"],
    "Groq": ["groq-gpt-001", "groq-gpt-002"],
    "DeepSeek": ["deepseek-gpt-001", "deepseek-gpt-002"]
}

# OpenAI model pricing per 1M tokens
OPENAI_PRICES = {
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
}

PROVIDER_TO_ENV_VAR_KEY = {
    "OpenAI": "OPENAI_API_KEY",
    "AzureOpenAI": "AZURE_OPENAI_API_KEY",
    "OpenRouter": "OPENROUTER_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Together": "TOGETHER_API_KEY",
    "Perplexity": "PERPLEXITY_API_KEY",
    "Groq": "GROQ_API_KEY",
    "DeepSeek": "DEEPSEEK_API_KEY",
}

# API Key validation functions
def validate_openai_key(api_key):
    """Validate OpenAI API key by making a minimal API call"""
    try:
        # First check if the API key is empty
        if not api_key:
            return False, "API key is empty"
            
        client = OpenAIClient(
            model_name="gpt-3.5-turbo",
            api_key=api_key,
            max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        
        # Catch authentication errors specifically
        try:
            client.chat(messages)
            return True, ""
        except Exception as e:
            if "401" in str(e) or "invalid_api_key" in str(e) or "Invalid API key" in str(e):
                return False, "Invalid API key. Please check your OpenAI API key."
            else:
                raise e
                
    except Exception as e:
        return False, str(e)


def validate_anthropic_key(api_key):
    """Validate Anthropic API key by making a minimal API call"""
    try:
        # First check if the API key is empty
        if not api_key:
            return False, "API key is empty"
            
        client = AnthropicClient(
            model_name="claude-3-haiku-20240307",
            api_key=api_key,
            max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        
        # Catch authentication errors specifically
        try:
            client.chat(messages)
            return True, ""
        except Exception as e:
            if "401" in str(e) or "invalid_api_key" in str(e) or "Invalid API key" in str(e):
                return False, "Invalid API key. Please check your Anthropic API key."
            else:
                raise e
                
    except Exception as e:
        return False, str(e)


def validate_together_key(api_key):
    """Validate Together API key by making a minimal API call"""
    try:
        # First check if the API key is empty
        if not api_key:
            return False, "API key is empty"
            
        client = TogetherClient(
            model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            api_key=api_key,
            max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        
        # Catch authentication errors specifically
        try:
            client.chat(messages)
            return True, ""
        except Exception as e:
            if "401" in str(e) or "invalid_api_key" in str(e) or "Invalid API key" in str(e):
                return False, "Invalid API key. Please check your Together API key."
            else:
                raise e
                
    except Exception as e:
        return False, str(e)


def validate_perplexity_key(api_key):
    """Validate Perplexity API key by making a minimal API call"""
    try:
        # First check if the API key is empty
        if not api_key:
            return False, "API key is empty"
            
        client = PerplexityAIClient(
            model_name="sonar-pro", 
            api_key=api_key, 
            max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        
        # Catch authentication errors specifically
        try:
            client.chat(messages)
            return True, ""
        except Exception as e:
            if "401" in str(e) or "invalid_api_key" in str(e) or "Invalid API key" in str(e):
                return False, "Invalid API key. Please check your Perplexity API key."
            else:
                raise e
                
    except Exception as e:
        return False, str(e)


def validate_openrouter_key(api_key):
    """Validate OpenRouter API key by making a minimal API call"""
    try:
        # First check if the API key is empty
        if not api_key:
            return False, "API key is empty"
            
        client = OpenRouterClient(
            model_name="anthropic/claude-3-5-sonnet",  # Use a common model for testing
            api_key=api_key,
            max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        
        # Catch authentication errors specifically
        try:
            client.chat(messages)
            return True, ""
        except Exception as e:
            if "401" in str(e) or "invalid_api_key" in str(e) or "Invalid API key" in str(e):
                return False, "Invalid API key. Please check your OpenRouter API key."
            else:
                raise e
                
    except Exception as e:
        return False, str(e)


def validate_azure_openai_key(api_key):
    """Validate Azure OpenAI API key by making a minimal API call"""
    try:
        # First check if the API key is empty
        if not api_key:
            return False, "API key is empty"
            
        client = AzureOpenAIClient(
            model_name="text-davinci-003",
            api_key=api_key,
            max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        
        # Catch authentication errors specifically
        try:
            client.chat(messages)
            return True, ""
        except Exception as e:
            if "401" in str(e) or "invalid_api_key" in str(e) or "Invalid API key" in str(e):
                return False, "Invalid API key. Please check your Azure OpenAI API key."
            else:
                raise e
                
    except Exception as e:
        return False, str(e)


def validate_groq_key(api_key):
    """Validate Groq API key by making a minimal API call"""
    try:
        # First check if the API key is empty
        if not api_key:
            return False, "API key is empty"
            
        client = GroqClient(
            model_name="groq-gpt-001",
            api_key=api_key,
            max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        
        # Catch authentication errors specifically
        try:
            client.chat(messages)
            return True, ""
        except Exception as e:
            if "401" in str(e) or "invalid_api_key" in str(e) or "Invalid API key" in str(e):
                return False, "Invalid API key. Please check your Groq API key."
            else:
                raise e
                
    except Exception as e:
        return False, str(e)


def validate_deepseek_key(api_key):
    """Validate DeepSeek API key by making a minimal API call"""
    try:
        # First check if the API key is empty
        if not api_key:
            return False, "API key is empty"
            
        client = DeepSeekClient(
            model_name="deepseek-gpt-001",
            api_key=api_key,
            max_tokens=1
        )
        messages = [{"role": "user", "content": "Say yes"}]
        
        # Catch authentication errors specifically
        try:
            client.chat(messages)
            return True, ""
        except Exception as e:
            if "401" in str(e) or "invalid_api_key" in str(e) or "Invalid API key" in str(e):
                return False, "Invalid API key. Please check your DeepSeek API key."
            else:
                raise e
                
    except Exception as e:
        return False, str(e)

# For Minions protocol
class JobOutput(BaseModel):
    answer: str | None
    explanation: str | None
    citation: str | None

class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None

class StructuredOutputSchema(BaseModel):
    answer: str
    explanation: str = None
    citation: str = None

class MinionsApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='com.hazyresearch.minions')
        
        # Initialize context window parameter
        self.num_ctx = 4096  # Default matching web client
        
        # Provider-related settings
        self.providers = ["OpenAI", "Anthropic", "Together", "Perplexity", "OpenRouter", "AzureOpenAI", "Groq", "DeepSeek"]
        self.current_provider = "OpenAI"
        self.api_key = OPENAI_API_KEY
        
        # Model settings
        self.local_model_name = "llama3.2:latest"  # Updated to match local model
        self.remote_model_name = "gpt-4o-mini"
        self.local_temperature = 0.7
        self.local_max_tokens = 1024
        self.remote_temperature = 0.7
        self.remote_max_tokens = 1024
        
        # Client instances
        self.local_client = None
        self.remote_client = None
        self.minion = None
        self.minions = None
        
        # Document management
        self.uploaded_docs = []
        self.doc_metadata = {}
        
        # MCP settings
        self.mcp_config_manager = MCPConfigManager()
        self.mcp_server_name = None
        
        # Privacy mode
        self.privacy_mode = False
        
    def do_activate(self):
        print("Activating application...")
        window = MainWindow(app=self)
        print("Window created, showing...")
        window.show_all()  # Explicitly show all widgets
        print("Window shown")
        
    def initialize_clients(self, local_model_name, remote_model_name, provider, protocol,
                          local_max_tokens, remote_max_tokens, api_key, num_ctx=4096, mcp_server_name=None):
        """Initialize the local and remote clients for the Minions protocol."""
        try:
            # Initialize local client (Ollama)
            try:
                # Define structured output schema for Ollama client
                structured_output_schema = StructuredOutputSchema if protocol == "Minions" else None
                
                self.local_client = OllamaClient(
                    model_name=local_model_name,
                    temperature=float(self.local_temperature),
                    max_tokens=int(local_max_tokens),
                    num_ctx=num_ctx,
                    structured_output_schema=structured_output_schema,
                    use_async=False  # Desktop client uses synchronous calls
                )
                print(f"Initialized local client with model {local_model_name}")
            except Exception as e:
                print(f"Warning: Could not initialize local client: {str(e)}")
                self.local_client = None
            
            # Initialize remote client based on provider
            if api_key:
                try:
                    if provider == "OpenAI":
                        self.remote_client = OpenAIClient(
                            model_name=remote_model_name,
                            api_key=api_key,
                            max_tokens=int(remote_max_tokens),
                            num_ctx=num_ctx,
                        )
                    elif provider == "Anthropic":
                        self.remote_client = AnthropicClient(
                            model_name=remote_model_name,
                            api_key=api_key,
                            max_tokens=int(remote_max_tokens)
                        )
                    elif provider == "Together":
                        self.remote_client = TogetherClient(
                            model_name=remote_model_name,
                            api_key=api_key,
                            max_tokens=int(remote_max_tokens)
                        )
                    elif provider == "Perplexity":
                        self.remote_client = PerplexityAIClient(
                            model_name=remote_model_name,
                            api_key=api_key,
                            max_tokens=int(remote_max_tokens)
                        )
                    elif provider == "OpenRouter":
                        self.remote_client = OpenRouterClient(
                            model_name=remote_model_name,
                            api_key=api_key,
                            max_tokens=int(remote_max_tokens)
                        )
                    elif provider == "AzureOpenAI":
                        self.remote_client = AzureOpenAIClient(
                            model_name=remote_model_name,
                            api_key=api_key,
                            max_tokens=int(remote_max_tokens)
                        )
                    elif provider == "Groq":
                        self.remote_client = GroqClient(
                            model_name=remote_model_name,
                            api_key=api_key,
                            max_tokens=int(remote_max_tokens)
                        )
                    elif provider == "DeepSeek":
                        self.remote_client = DeepSeekClient(
                            model_name=remote_model_name,
                            api_key=api_key,
                            max_tokens=int(remote_max_tokens)
                        )
                    print(f"Initialized remote client with provider {provider} and model {remote_model_name}")
                except Exception as e:
                    print(f"Warning: Could not initialize remote client: {str(e)}")
                    self.remote_client = None
            
            # Initialize protocol
            if protocol == "Minion":
                if self.local_client:
                    self.minion = Minion(self.local_client)
                    print("Initialized Minion protocol with local client")
                    return True, "Initialized Minion protocol with local client"
                elif self.remote_client:
                    self.minion = Minion(self.remote_client)
                    print("Initialized Minion protocol with remote client")
                    return True, "Initialized Minion protocol with remote client"
                else:
                    return False, "No clients available for Minion protocol"
            elif protocol == "Minions":
                if self.local_client and self.remote_client:
                    self.minions = Minions(self.local_client, self.remote_client)
                    print("Initialized Minions protocol with both clients")
                    return True, "Initialized Minions protocol with both clients"
                elif self.local_client:
                    # Fallback to using only local client
                    self.minions = Minions(self.local_client, self.local_client)
                    print("Initialized Minions protocol with local client only (fallback)")
                    return True, "Initialized Minions protocol with local client only (remote client not available)"
                elif self.remote_client:
                    # Fallback to using only remote client
                    self.minions = Minions(self.remote_client, self.remote_client)
                    print("Initialized Minions protocol with remote client only (fallback)")
                    return True, "Initialized Minions protocol with remote client only (local client not available)"
                else:
                    return False, "No clients available for Minions protocol"
            elif protocol == "Minions-MCP":
                if self.local_client and self.remote_client and mcp_server_name:
                    self.minions = SyncMinionsMCP(
                        local_client=self.local_client,
                        remote_client=self.remote_client,
                        mcp_server_name=mcp_server_name
                    )
                    print(f"Initialized Minions-MCP protocol with both clients and server {mcp_server_name}")
                    return True, f"Initialized Minions-MCP protocol with both clients and server {mcp_server_name}"
                else:
                    return False, "Missing clients or MCP server name for Minions-MCP protocol"
            else:
                return False, f"Unknown protocol: {protocol}"
        except Exception as e:
            return False, str(e)

class MCPServerConfig:
    """Configuration for an MCP server"""
    
    def __init__(self, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        """Initialize MCP server configuration
        
        Args:
            command: Command to run the MCP server
            args: Arguments to pass to the command
            env: Environment variables to set when running the command
        """
        self.command = command
        self.args = args
        self.env = env or {}

class MCPConfigManager:
    """Manages MCP server configurations"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize MCP config manager

        Args:
            config_path: Path to MCP config file. If None, will look in default locations
        """
        self.config_path = config_path
        self.servers: Dict[str, MCPServerConfig] = {}
        self._load_config()

    def _load_config(self):
        """Load MCP configuration from file"""
        paths_to_try = [
            self.config_path,
            os.path.join(os.getcwd(), "mcp.json"),
            os.path.join(os.getcwd(), ".mcp.json"),
            os.path.expanduser("~/.mcp.json"),
        ]

        config_file = None
        for path in paths_to_try:
            if path and os.path.exists(path):
                config_file = path
                break

        if not config_file:
            return

        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            if "mcpServers" in config:
                for server_name, server_config in config["mcpServers"].items():
                    self.servers[server_name] = MCPServerConfig(
                        command=server_config["command"],
                        args=server_config["args"],
                        env=server_config.get("env"),
                    )
        except Exception as e:
            raise ValueError(f"Failed to load MCP config from {config_file}: {str(e)}")

    def get_server_config(self, server_name: str) -> MCPServerConfig:
        """Get configuration for a specific MCP server"""
        if server_name not in self.servers:
            raise ValueError(f"MCP server '{server_name}' not found in config")
        return self.servers[server_name]

    def list_servers(self) -> list[str]:
        """Get list of configured server names"""
        return list(self.servers.keys())
    
    def get_servers(self) -> list[str]:
        """Get list of configured server names (alias for list_servers)"""
        return self.list_servers()

class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(title="Minions Desktop Client")
        self.props.application = app
        self.set_default_size(800, 600)
        
        # Initialize chat messages list
        self.chat_messages = []
        
        # Default protocol
        self.protocol = "Minion"
        
        # Main paned container
        self.main_paned = Gtk.Paned.new(Gtk.Orientation.HORIZONTAL)
        self.add(self.main_paned)
        print("Added main paned container")
        
        # Protocol settings
        self.protocol_options = ["Minion", "Minions", "Minions-MCP"]
        
        # Chat message storage
        self.chat_messages = []
        
        # Initialize components
        self.build_sidebar()
        self.build_main_content()
        self.setup_branding()
        
        print("All components built")

    def build_sidebar(self):
        print("Building sidebar...")
        self.sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.sidebar.set_margin_start(10)
        self.sidebar.set_margin_end(10)
        self.sidebar.set_margin_top(10)
        self.main_paned.add1(self.sidebar)
        print("Sidebar added to paned container")

        # Provider selection
        provider_label = Gtk.Label(label="AI Provider:")
        self.provider_combo = Gtk.ComboBoxText()
        for p in self.props.application.providers:
            self.provider_combo.append_text(p)
        self.provider_combo.set_active(0)
        self.provider_combo.connect("changed", self.on_provider_changed)

        # API Key entry
        api_key_label = Gtk.Label(label="API Key:")
        self.api_key_entry = Gtk.Entry()
        self.api_key_entry.set_visibility(False)
        self.api_key_entry.connect("changed", self.on_api_key_changed)

        # Model selection
        model_label = Gtk.Label(label="Model:")
        self.model_combo = Gtk.ComboBoxText()
        self.update_models()
        
        # Protocol selection
        protocol_label = Gtk.Label(label="Protocol:")
        self.protocol_combo = Gtk.ComboBoxText()
        for protocol in self.protocol_options:
            self.protocol_combo.append_text(protocol)
        self.protocol_combo.set_active(0)  # Default to Minion
        self.protocol = "Minion"  # Initialize protocol variable
        self.protocol_combo.connect("changed", self.on_protocol_changed)
        
        # MCP Server selection
        self.mcp_server_label = Gtk.Label(label="MCP Server:")
        self.mcp_server_combo = Gtk.ComboBoxText()
        self.update_mcp_servers()
        self.mcp_server_combo.connect("changed", self.on_mcp_server_changed)
        
        # Set initial visibility of MCP server controls (hidden by default)
        mcp_server_visible = self.protocol == "Minions-MCP"
        self.mcp_server_combo.set_visible(mcp_server_visible)
        self.mcp_server_label.set_visible(mcp_server_visible)
        self.mcp_server_combo.set_no_show_all(not mcp_server_visible)
        self.mcp_server_label.set_no_show_all(not mcp_server_visible)
        
        # Force UI update to properly hide widgets if needed
        if not mcp_server_visible:
            self.mcp_server_combo.hide()
            self.mcp_server_label.hide()
        
        # Privacy Mode toggle
        privacy_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        privacy_label = Gtk.Label(label="Privacy Mode:")
        self.privacy_switch = Gtk.Switch()
        self.privacy_switch.set_active(False)
        self.privacy_switch.connect("notify::active", self.on_privacy_toggled)
        privacy_box.pack_start(privacy_label, False, False, 0)
        privacy_box.pack_start(self.privacy_switch, False, False, 0)
        
        # Local model settings
        local_model_label = Gtk.Label(label="Local Model:")
        self.local_model_entry = Gtk.Entry()
        self.local_model_entry.set_text(self.props.application.local_model_name)
        self.local_model_entry.connect("changed", self.on_local_model_changed)
        
        # Temperature settings
        temp_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        temp_label = Gtk.Label(label="Temperature:")
        self.temp_adjustment = Gtk.Adjustment(value=0.7, lower=0.0, upper=1.0, step_increment=0.1)
        self.temp_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.temp_adjustment)
        self.temp_scale.set_digits(1)
        self.temp_scale.connect("value-changed", self.on_temperature_changed)
        temp_box.pack_start(temp_label, False, False, 0)
        temp_box.pack_start(self.temp_scale, True, True, 0)

        # Document upload
        upload_btn = Gtk.Button(label="Upload Document")
        upload_btn.connect("clicked", self.on_upload_clicked)

        # Assembly
        self.sidebar.pack_start(provider_label, False, False, 0)
        self.sidebar.pack_start(self.provider_combo, False, False, 0)
        self.sidebar.pack_start(api_key_label, False, False, 0)
        self.sidebar.pack_start(self.api_key_entry, False, False, 0)
        self.sidebar.pack_start(model_label, False, False, 0)
        self.sidebar.pack_start(self.model_combo, False, False, 0)
        self.sidebar.pack_start(protocol_label, False, False, 0)
        self.sidebar.pack_start(self.protocol_combo, False, False, 0)
        self.sidebar.pack_start(self.mcp_server_label, False, False, 0)
        self.sidebar.pack_start(self.mcp_server_combo, False, False, 0)
        self.sidebar.pack_start(privacy_box, False, False, 0)
        self.sidebar.pack_start(local_model_label, False, False, 0)
        self.sidebar.pack_start(self.local_model_entry, False, False, 0)
        self.sidebar.pack_start(temp_box, False, False, 0)
        self.sidebar.pack_start(upload_btn, False, False, 5)
        self.build_document_section()
        print("Sidebar components created and packed")

    def update_mcp_servers(self):
        self.mcp_server_combo.remove_all()
        for server in self.props.application.mcp_config_manager.get_servers():
            self.mcp_server_combo.append_text(server)
        self.mcp_server_combo.set_active(0)

    def build_main_content(self):
        print("Building main content...")
        self.main_content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.main_paned.add2(self.main_content)
        print("Main content added to paned container")

        # Create the chat interface
        self.chat_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.main_content.pack_start(self.chat_box, True, True, 0)
        
        # Chat history
        scrolled = Gtk.ScrolledWindow()
        self.chat_history = Gtk.TextView()
        self.chat_history.set_editable(False)
        self.chat_history.set_wrap_mode(Gtk.WrapMode.WORD)
        scrolled.add(self.chat_history)

        # Input area
        input_box = Gtk.Box(spacing=5)
        self.chat_input = Gtk.Entry()
        self.chat_input.connect("activate", self.on_send_message)  # Handle Enter key press
        send_btn = Gtk.Button(label="Send")
        send_btn.connect("clicked", self.on_send_message)

        input_box.pack_start(self.chat_input, True, True, 0)
        input_box.pack_start(send_btn, False, False, 0)

        self.chat_box.pack_start(scrolled, True, True, 0)
        self.chat_box.pack_start(input_box, False, False, 5)
        print("Chat components created and packed")

    def build_document_section(self):
        print("Building document section...")
        self.doc_list = Gtk.ListBox()
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_min_content_height(150)
        scrolled.add(self.doc_list)
        
        self.sidebar.pack_start(Gtk.Separator(), False, False, 5)
        self.sidebar.pack_start(scrolled, True, True, 0)
        print("Document section created and packed")

    def setup_branding(self):
        print("Setting up branding...")
        display = self.get_display()
        scale_factor = display.get_monitor(0).get_scale_factor()
        dark_mode = self.is_dark_mode()

        image_path = os.path.abspath(
            "assets/minions_logo_no_background.png" if dark_mode 
            else "assets/minions_logo_light.png"
        )
        print(f"Loading logo from: {image_path}")

        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(image_path)
            scaled_pixbuf = pixbuf.scale_simple(
                int(200 * scale_factor),
                int((pixbuf.get_height() * 200 / pixbuf.get_width()) * scale_factor),
                GdkPixbuf.InterpType.HYPER
            )
            logo = Gtk.Image.new_from_pixbuf(scaled_pixbuf)
            
            branding_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            branding_box.set_halign(Gtk.Align.CENTER)
            branding_box.pack_start(logo, False, False, 0)
            self.main_content.pack_start(branding_box, False, False, 0)
            print("Logo loaded and added successfully")

        except Exception as e:
            print(f"Error loading logo: {str(e)}")
            error_label = Gtk.Label(label=f"Error loading logo: {str(e)}")
            self.main_content.pack_start(error_label, False, False, 0)

    def is_dark_mode(self):
        settings = Gtk.Settings.get_default()
        return settings.get_property("gtk-application-prefer-dark-theme")

    def add_message_to_chat(self, sender, message, is_thinking=False):
        """Add a message to the chat display with proper formatting."""
        # Add to the text buffer first (for backward compatibility)
        buffer = self.chat_history.get_buffer()
        
        # Get the current end position before inserting text
        start_mark = buffer.create_mark(None, buffer.get_end_iter(), True)
        
        # Format the message for text buffer display
        if sender == "You":
            buffer.insert(buffer.get_end_iter(), f"{sender}: {message}\n")
        elif sender == "Assistant":
            if is_thinking:
                # Add a thinking indicator with animation dots
                buffer.insert(buffer.get_end_iter(), f"AI: Thinking")
                # Start the thinking animation
                self.thinking_dots_count = 0
                GLib.timeout_add(500, self.animate_thinking_dots)
            elif isinstance(message, (dict, list)) or (isinstance(message, str) and message.strip().startswith('{')):
                # Format structured output - handle both dict/list and JSON strings
                try:
                    # Try to parse as JSON if it's a string
                    if isinstance(message, str) and message.strip().startswith('{'):
                        try:
                            message = json.loads(message)
                        except json.JSONDecodeError:
                            pass  # Keep as string if not valid JSON
                    
                    # Now format the structured output
                    formatted_message = format_structured_output(message)
                    buffer.insert(buffer.get_end_iter(), f"AI: {formatted_message}\n\n")
                except Exception as e:
                    print(f"Error formatting structured output: {e}")
                    buffer.insert(buffer.get_end_iter(), f"AI: {message}\n\n")
            else:
                buffer.insert(buffer.get_end_iter(), f"AI: {message}\n\n")
        else:
            buffer.insert(buffer.get_end_iter(), f"{sender}: {message}\n")
        
        # Get the end position after inserting text
        end_iter = buffer.get_end_iter()
        start_iter = buffer.get_iter_at_mark(start_mark)
        
        # Add CSS styling for different message types
        if sender == "You":
            tag = buffer.create_tag(None)
            tag.set_property("foreground", "blue")
            tag.set_property("weight", 600)
            buffer.apply_tag(tag, start_iter, end_iter)
        elif sender == "Assistant":
            tag = buffer.create_tag(None)
            tag.set_property("foreground", "green")
            tag.set_property("weight", 600)
            buffer.apply_tag(tag, start_iter, end_iter)
        else:
            tag = buffer.create_tag(None)
            tag.set_property("foreground", "red")
            tag.set_property("weight", 600)
            buffer.apply_tag(tag, start_iter, end_iter)
        
        # Clean up the mark
        buffer.delete_mark(start_mark)
            
        # Store the message for later reference
        self.chat_messages.append({
            'sender': sender,
            'message': message,
            'is_thinking': is_thinking
        })
        
        # Ensure the chat history scrolls to show the latest message
        adj = self.chat_history.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())

    def remove_thinking_message(self):
        """Remove the thinking message from the chat."""
        # Stop the animation
        if hasattr(self, 'thinking_dots_count'):
            delattr(self, 'thinking_dots_count')
            
        # Find and remove the last message if it's a thinking message
        for i in range(len(self.chat_messages) - 1, -1, -1):
            if self.chat_messages[i].get('is_thinking', False):
                # Remove from the list
                self.chat_messages.pop(i)
                # Remove from the text buffer
                buffer = self.chat_history.get_buffer()
                start_iter = buffer.get_start_iter()
                end_iter = buffer.get_end_iter()
                buffer.delete(start_iter, end_iter)
                
                # Redraw all messages except the thinking one
                for msg in self.chat_messages:
                    if sender := msg.get('sender'):
                        message = msg.get('message', '')
                        is_thinking = msg.get('is_thinking', False)
                        
                        # Skip thinking messages
                        if not is_thinking:
                            # Format the message for text buffer display
                            if sender == "You":
                                buffer.insert(buffer.get_end_iter(), f"{sender}: {message}\n")
                            elif sender == "Assistant":
                                if isinstance(message, (dict, list)) or (isinstance(message, str) and message.strip().startswith('{')):
                                    # Format structured output - handle both dict/list and JSON strings
                                    try:
                                        # Try to parse as JSON if it's a string
                                        if isinstance(message, str) and message.strip().startswith('{'):
                                            try:
                                                message = json.loads(message)
                                            except json.JSONDecodeError:
                                                pass  # Keep as string if not valid JSON
                    
                                        # Now format the structured output
                                        formatted_message = format_structured_output(message)
                                        buffer.insert(buffer.get_end_iter(), f"AI: {formatted_message}\n\n")
                                    except Exception as e:
                                        print(f"Error formatting structured output: {e}")
                                        buffer.insert(buffer.get_end_iter(), f"AI: {message}\n\n")
                                else:
                                    buffer.insert(buffer.get_end_iter(), f"AI: {message}\n\n")
                            else:
                                buffer.insert(buffer.get_end_iter(), f"{sender}: {message}\n")
                
                # Ensure the chat history scrolls to show the latest message
                adj = self.chat_history.get_vadjustment()
                adj.set_value(adj.get_upper() - adj.get_page_size())

        return False  # Return False to stop the idle_add callback

    def animate_thinking_dots(self):
        """Animate the thinking dots for the AI response."""
        if not hasattr(self, 'thinking_dots_count'):
            return False
            
        # Find the last message if it's a thinking message
        for i in range(len(self.chat_messages) - 1, -1, -1):
            if self.chat_messages[i].get('is_thinking', False):
                buffer = self.chat_history.get_buffer()
                
                # Get the end position of the buffer
                end_iter = buffer.get_end_iter()
                
                # Delete any existing dots
                line_start = buffer.get_iter_at_line(buffer.get_line_count() - 1)
                buffer.delete(line_start, end_iter)
                
                # Add new dots based on the counter
                dots = "." * ((self.thinking_dots_count % 3) + 1)
                buffer.insert(buffer.get_end_iter(), f"Thinking{dots}")
                
                # Increment the counter
                self.thinking_dots_count += 1
                
                # Continue the animation
                return True
                
        # If we didn't find a thinking message, stop the animation
        return False

    def on_mcp_server_changed(self, combo):
        app = self.props.application
        server_name = combo.get_active_text()
        app.mcp_server_name = server_name
        
        # Reinitialize clients with the new MCP server
        try:
            success, message = app.initialize_clients(
                local_model_name=app.local_model_name,
                remote_model_name=app.remote_model_name,
                provider=app.current_provider,
                protocol=self.protocol,
                local_max_tokens=app.local_max_tokens,
                remote_max_tokens=app.remote_max_tokens,
                api_key=app.api_key,
                num_ctx=app.num_ctx,
                mcp_server_name=server_name
            )
            
            if not success:
                warning_dialog = Gtk.MessageDialog(
                    transient_for=self,
                    modal=True,
                    message_type=Gtk.MessageType.WARNING,
                    buttons=Gtk.ButtonsType.OK,
                    text=f"Note: {message}\n\nYou may need to enter a valid API key."
                )
                warning_dialog.run()
                warning_dialog.destroy()
        except Exception as e:
            # Handle any exceptions during client initialization
            error_dialog = Gtk.MessageDialog(
                transient_for=self,
                modal=True,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text=f"Error initializing clients: {str(e)}"
            )
            error_dialog.run()
            error_dialog.destroy()

    def on_privacy_toggled(self, switch, gparam):
        app = self.props.application
        app.privacy_mode = switch.get_active()
        
        # Update UI based on privacy mode
        buffer = self.chat_history.get_buffer()
        buffer.insert(buffer.get_end_iter(), f"System: Switched to {'Private' if app.privacy_mode else 'Public'} mode\n")
        
        # Ensure the chat history scrolls to show the latest message
        adj = self.chat_history.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())

    def on_provider_changed(self, combo):
        app = self.props.application
        provider = combo.get_active_text()
        app.current_provider = provider
        
        # Set API key from environment variable based on selected provider
        if provider == "OpenAI":
            app.api_key = OPENAI_API_KEY
        elif provider == "Anthropic":
            app.api_key = ANTHROPIC_API_KEY
        elif provider == "Together":
            app.api_key = TOGETHER_API_KEY
        elif provider == "Perplexity":
            app.api_key = PERPLEXITY_API_KEY
        elif provider == "OpenRouter":
            app.api_key = OPENROUTER_API_KEY
        elif provider == "AzureOpenAI":
            app.api_key = AZURE_OPENAI_API_KEY
        elif provider == "Groq":
            app.api_key = GROQ_API_KEY
        elif provider == "DeepSeek":
            app.api_key = DEEPSEEK_API_KEY
            
        # Update the API key entry field with the value from environment
        if app.api_key:
            self.api_key_entry.set_text(app.api_key)
            
            # Only validate if the user explicitly wants to
            # We'll skip automatic validation when changing providers to avoid errors
            # The user can validate manually by changing the API key in the entry field
        else:
            # No API key available, show warning
            warning_dialog = Gtk.MessageDialog(
                transient_for=self,
                modal=True,
                message_type=Gtk.MessageType.WARNING,
                buttons=Gtk.ButtonsType.OK,
                text=f"No API key found for {provider}. Please enter an API key."
            )
            warning_dialog.run()
            warning_dialog.destroy()
            
        self.update_models()
        
        # Reinitialize clients with the new provider
        try:
            success, message = app.initialize_clients(
                local_model_name=app.local_model_name,
                remote_model_name=app.remote_model_name,
                provider=provider,
                protocol=self.protocol,
                local_max_tokens=app.local_max_tokens,
                remote_max_tokens=app.remote_max_tokens,
                api_key=app.api_key,
                num_ctx=app.num_ctx
            )
            
            if not success:
                warning_dialog = Gtk.MessageDialog(
                    transient_for=self,
                    modal=True,
                    message_type=Gtk.MessageType.WARNING,
                    buttons=Gtk.ButtonsType.OK,
                    text=f"Note: {message}\n\nYou may need to enter a valid API key."
                )
                warning_dialog.run()
                warning_dialog.destroy()
        except Exception as e:
            # Handle any exceptions during client initialization
            error_dialog = Gtk.MessageDialog(
                transient_for=self,
                modal=True,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text=f"Error initializing clients: {str(e)}"
            )
            error_dialog.run()
            error_dialog.destroy()

    def update_models(self):
        self.model_combo.remove_all()
        provider = self.provider_combo.get_active_text()
        for model in MODEL_MAP.get(provider, []):
            self.model_combo.append_text(model)
        self.model_combo.set_active(0)

    def on_api_key_changed(self, entry):
        """Update API key when entry changes"""
        app = self.props.application
        api_key = entry.get_text()
        app.api_key = api_key
        
        # Validate the API key based on the selected provider
        provider = app.current_provider
        is_valid = False
        message = ""
        
        if api_key:
            if provider == "OpenAI":
                is_valid, message = validate_openai_key(api_key)
            elif provider == "Anthropic":
                is_valid, message = validate_anthropic_key(api_key)
            elif provider == "Together":
                is_valid, message = validate_together_key(api_key)
            elif provider == "Perplexity":
                is_valid, message = validate_perplexity_key(api_key)
            elif provider == "OpenRouter":
                is_valid, message = validate_openrouter_key(api_key)
            elif provider == "AzureOpenAI":
                is_valid, message = validate_azure_openai_key(api_key)
            elif provider == "Groq":
                is_valid, message = validate_groq_key(api_key)
            elif provider == "DeepSeek":
                is_valid, message = validate_deepseek_key(api_key)
            
            # Show validation result to the user
            if is_valid:
                info_dialog = Gtk.MessageDialog(
                    transient_for=self,
                    modal=True,
                    message_type=Gtk.MessageType.INFO,
                    buttons=Gtk.ButtonsType.OK,
                    text="API key is valid. You're good to go!"
                )
                info_dialog.run()
                info_dialog.destroy()
            else:
                error_dialog = Gtk.MessageDialog(
                    transient_for=self,
                    modal=True,
                    message_type=Gtk.MessageType.ERROR,
                    buttons=Gtk.ButtonsType.OK,
                    text=f"Invalid API key: {message}"
                )
                error_dialog.run()
                error_dialog.destroy()

    def on_protocol_changed(self, combo):
        """Handle protocol selection change."""
        self.protocol = combo.get_active_text()
        print(f"Protocol changed to: {self.protocol}")
        
        # Show/hide MCP server selection based on protocol
        mcp_server_visible = (self.protocol == "Minions-MCP")
        self.mcp_server_combo.set_visible(mcp_server_visible)
        self.mcp_server_label.set_visible(mcp_server_visible)
        self.mcp_server_combo.set_no_show_all(not mcp_server_visible)
        self.mcp_server_label.set_no_show_all(not mcp_server_visible)
        
        # Force UI update to properly show/hide widgets
        if mcp_server_visible:
            self.mcp_server_combo.show_all()
            self.mcp_server_label.show_all()
        else:
            self.mcp_server_combo.hide()
            self.mcp_server_label.hide()
        
        # Update UI based on protocol
        buffer = self.chat_history.get_buffer()
        buffer.insert(buffer.get_end_iter(), f"System: Switched to {self.protocol} protocol\n")
        
        # Ensure the chat history scrolls to show the latest message
        adj = self.chat_history.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())
        
        # Reinitialize clients with the new protocol
        app = self.props.application
        try:
            success, message = app.initialize_clients(
                local_model_name=app.local_model_name,
                remote_model_name=app.remote_model_name,
                provider=app.current_provider,
                protocol=self.protocol,
                local_max_tokens=app.local_max_tokens,
                remote_max_tokens=app.remote_max_tokens,
                api_key=app.api_key,
                num_ctx=app.num_ctx,
                mcp_server_name=app.mcp_server_name if self.protocol == "Minions-MCP" else None
            )
            
            if not success:
                warning_dialog = Gtk.MessageDialog(
                    transient_for=self,
                    modal=True,
                    message_type=Gtk.MessageType.WARNING,
                    buttons=Gtk.ButtonsType.OK,
                    text=f"Note: {message}\n\nYou may need to enter a valid API key."
                )
                warning_dialog.run()
                warning_dialog.destroy()
        except Exception as e:
            # Handle any exceptions during client initialization
            error_dialog = Gtk.MessageDialog(
                transient_for=self,
                modal=True,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text=f"Error initializing clients: {str(e)}"
            )
            error_dialog.run()
            error_dialog.destroy()

    def on_local_model_changed(self, entry):
        """Update local model when entry changes"""
        self.props.application.local_model_name = entry.get_text()
        
    def on_temperature_changed(self, scale):
        """Update temperature settings when scale changes"""
        value = scale.get_value()
        self.props.application.local_temperature = value
        self.props.application.remote_temperature = value
        
    def on_send_message(self, widget):
        message = self.chat_input.get_text()
        if not message:
            return
        
        # Add user message to chat
        self.add_message_to_chat("You", message)
        self.chat_input.set_text("")
        
        # Execute the appropriate protocol
        self.run_protocol(message)

    def run_protocol(self, task, context=None):
        """Run the selected protocol with the given task"""
        import threading
        from gi.repository import GLib
        
        # Add a "thinking" message
        self.add_message_to_chat("Assistant", "Thinking...", is_thinking=True)
        
        def _run():
            app = self.props.application
            
            # Desktop-style client initialization
            if not app.local_client:
                # Define structured output schema for Ollama client
                structured_output_schema = StructuredOutputSchema if self.protocol == "Minions" else None
                
                app.local_client = OllamaClient(
                    model_name=app.local_model_name,
                    temperature=float(app.local_temperature),
                    max_tokens=int(app.local_max_tokens),
                    num_ctx=int(app.num_ctx),
                    structured_output_schema=structured_output_schema,
                    use_async=False  # Desktop client uses synchronous calls
                )
            
            # Prepare document context if available
            doc_context = ""
            doc_metadata = {}
            for doc in app.uploaded_docs:
                doc_context += f"\n\n{doc['text']}"
                doc_metadata[doc['name']] = doc['metadata']
            
            try:
                # Use the appropriate protocol
                if self.protocol == "Minions" and app.minions and app.remote_client:
                    # Use Minions protocol with both clients
                    output = app.minions(
                        task=task,
                        doc_metadata=doc_metadata,
                        context=[doc_context] if doc_context else None,
                        max_rounds=5,
                        is_privacy=app.privacy_mode
                    )
                    
                    # Remove the thinking message and add the response
                    GLib.idle_add(self.remove_thinking_message)
                    GLib.idle_add(self.add_message_to_chat, "Assistant", output)
                
                elif self.protocol == "Minions-MCP" and app.minions and app.remote_client:
                    # Use Minions-MCP protocol with both clients
                    output = app.minions(
                        task=task,
                        doc_metadata=doc_metadata,
                        context=[doc_context] if doc_context else None,
                        max_rounds=5,
                        is_privacy=app.privacy_mode
                    )
                    
                    # Remove the thinking message and add the response
                    GLib.idle_add(self.remove_thinking_message)
                    GLib.idle_add(self.add_message_to_chat, "Assistant", output)
                
                elif self.protocol == "Minion" and app.minion:
                    # Use Minion protocol
                    output = app.minion(
                        task=task,
                        doc_metadata=doc_metadata,
                        context=[doc_context] if doc_context else None,
                        max_rounds=5,
                        is_privacy=app.privacy_mode
                    )
                    
                    # Remove the thinking message and add the response
                    GLib.idle_add(self.remove_thinking_message)
                    GLib.idle_add(self.add_message_to_chat, "Assistant", output)
                
                # Fallback to direct client usage if protocol not initialized
                elif app.remote_client:
                    # Include document context in the prompt
                    full_prompt = task
                    if doc_context:
                        full_prompt = f"Context:\n{doc_context}\n\nQuestion: {task}"
                    
                    remote_responses, remote_usage, _ = app.remote_client.chat(
                        messages=[{"role": "user", "content": full_prompt}]
                    )
                    
                    # Remove the thinking message and add the response
                    GLib.idle_add(self.remove_thinking_message)
                    GLib.idle_add(self.add_message_to_chat, "Assistant", remote_responses[0])
                
                # Fallback to local client only
                else:
                    # Include document context in the prompt
                    full_prompt = task
                    if doc_context:
                        full_prompt = f"Context:\n{doc_context}\n\nQuestion: {task}"
                    
                    responses, usage, _ = app.local_client.chat(
                        messages=[{"role": "user", "content": full_prompt}]
                    )
                    
                    # Remove the thinking message and add the response
                    GLib.idle_add(self.remove_thinking_message)
                    GLib.idle_add(self.add_message_to_chat, "Assistant", responses[0])
                
            except Exception as e:
                # Remove the thinking message
                GLib.idle_add(self.remove_thinking_message)
                
                # Show error message
                error_message = f"Protocol error: {str(e)}"
                GLib.idle_add(self.add_message_to_chat, "System", error_message)
                
                # Show error dialog
                def show_error_dialog():
                    error_dialog = Gtk.MessageDialog(
                        transient_for=self,
                        modal=True,
                        message_type=Gtk.MessageType.ERROR,
                        buttons=Gtk.ButtonsType.OK,
                        text=f"Protocol error: {str(e)}"
                    )
                    error_dialog.run()
                    error_dialog.destroy()
                GLib.idle_add(show_error_dialog)
        
        # Desktop-specific thread handling
        threading.Thread(target=_run, daemon=True).start()

    def on_upload_clicked(self, widget):
        dialog = Gtk.FileChooserNative(
            title="Choose Documents",
            transient_for=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.connect("response", self.on_file_selected)
        dialog.show()

    def on_file_selected(self, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            file_path = dialog.get_filename()
            if not file_path:
                return
                
            try:
                # Process the file based on its extension
                file_name = os.path.basename(file_path)
                extension = os.path.splitext(file_name)[1].lower()
                
                # Read file content
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                
                # Extract text based on file type
                text = None
                if extension == ".pdf":
                    text = extract_text_from_pdf(file_bytes)
                    if not text:
                        raise Exception("Failed to extract text from PDF. Make sure PyMuPDF is installed.")
                elif extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
                    text = extract_text_from_image(file_bytes, self)
                    if not text:
                        raise Exception("Failed to extract text from image. Make sure pytesseract is installed.")
                elif extension in [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv", ".yml", ".yaml"]:
                    text = file_bytes.decode("utf-8", errors="replace")
                elif extension in [".docx", ".doc"]:
                    try:
                        import docx
                        from io import BytesIO
                        doc = docx.Document(BytesIO(file_bytes))
                        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    except ImportError:
                        raise Exception("python-docx library is required to process Word documents.")
                else:
                    # Try to decode as text
                    try:
                        text = file_bytes.decode("utf-8", errors="replace")
                    except:
                        text = f"[Binary file: {file_name}]"
                
                if not text or text.strip() == "":
                    text = f"[Empty or unprocessable file: {file_name}]"
                
                # Add to uploaded documents
                doc_info = {
                    "name": file_name,
                    "path": file_path,
                    "size": len(file_bytes),
                    "text": text,
                    "metadata": {
                        "filename": file_name,
                        "filepath": file_path,
                        "filesize": len(file_bytes),
                        "filetype": extension[1:] if extension else "unknown"
                    }
                }
                
                self.props.application.uploaded_docs.append(doc_info)
                
                # Update document list
                row = Gtk.ListBoxRow()
                hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
                
                # Add icon based on file type
                icon_name = "text-x-generic"
                if extension == ".pdf":
                    icon_name = "application-pdf"
                elif extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                    icon_name = "image-x-generic"
                elif extension in [".docx", ".doc"]:
                    icon_name = "x-office-document"
                
                icon = Gtk.Image.new_from_icon_name(icon_name, Gtk.IconSize.MENU)
                hbox.pack_start(icon, False, False, 0)
                
                # Add filename
                label = Gtk.Label(label=file_name)
                hbox.pack_start(label, True, True, 0)
                
                # Add file size
                size_str = self.format_file_size(len(file_bytes))
                size_label = Gtk.Label(label=size_str)
                size_label.set_alignment(1, 0.5)  # Right-align
                hbox.pack_end(size_label, False, False, 5)
                
                row.add(hbox)
                self.doc_list.add(row)
                self.doc_list.show_all()
                
                # Show success message
                success_dialog = Gtk.MessageDialog(
                    transient_for=self,
                    modal=True,
                    message_type=Gtk.MessageType.INFO,
                    buttons=Gtk.ButtonsType.OK,
                    text=f"Successfully processed {file_name}"
                )
                success_dialog.run()
                success_dialog.destroy()
                
            except Exception as e:
                error_dialog = Gtk.MessageDialog(
                    transient_for=self,
                    modal=True,
                    message_type=Gtk.MessageType.ERROR,
                    buttons=Gtk.ButtonsType.OK,
                    text=f"Error processing file: {str(e)}"
                )
                error_dialog.run()
                error_dialog.destroy()
                
    def format_file_size(self, size_bytes):
        """Format file size in human-readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def format_structured_output(output):
    """Format structured output for display in the chat window."""
    if isinstance(output, list):
        # For Minions protocol, messages are a list of jobs
        result = "Here are the outputs from all the minions:\n\n"
        
        # Group jobs by task_id
        tasks = {}
        for job in output:
            task_id = job.manifest.task_id
            if task_id not in tasks:
                tasks[task_id] = {"task": job.manifest.task, "jobs": []}
            tasks[task_id]["jobs"].append(job)
        
        for task_id, task_info in tasks.items():
            # Sort jobs by job_id
            task_info["jobs"] = sorted(
                task_info["jobs"], key=lambda x: x.manifest.job_id
            )
            
            # Filter jobs that have relevant information
            include_jobs = [
                job
                for job in task_info["jobs"]
                if job.output.answer
                and job.output.answer.lower().strip() != "none"
            ]
            
            result += f"Note: {len(task_info['jobs']) - len(include_jobs)} jobs did not have relevant information.\n\n"
            result += "Jobs with relevant information:\n\n"
            
            # Print all the relevant information
            for job in include_jobs:
                result += f"✅ Job {job.manifest.job_id + 1} (Chunk {job.manifest.chunk_id + 1})\n"
                result += f"Answer: {job.output.answer}\n\n"
                if hasattr(job.output, 'explanation') and job.output.explanation:
                    result += f"Explanation: {job.output.explanation}\n\n"
                if hasattr(job.output, 'citation') and job.output.citation:
                    result += f"Citation: {job.output.citation}\n\n"
        
        return result
    
    elif isinstance(output, dict):
        # Handle structured output from OllamaClient or direct JSON
        if "answer" in output:
            result = f"{output['answer']}\n\n"
            
            if "explanation" in output and output["explanation"]:
                result += f"{output['explanation']}\n\n"
                
            if "citation" in output and output["citation"]:
                result += f"Citation: {output['citation']}\n\n"
                
            return result
        # Try to handle dictionary output (e.g., JSON)
        elif "content" in output and isinstance(output["content"], (dict, str)):
            try:
                # Try to parse as JSON if it's a string
                if isinstance(output["content"], str) and output["content"].strip().startswith('{'):
                    try:
                        output["content"] = json.loads(output["content"])
                    except json.JSONDecodeError:
                        pass  # Keep as string if not valid JSON
                    
                # Now format the structured output
                formatted_message = format_structured_output(output["content"])
                return formatted_message
            except Exception as e:
                print(f"Error formatting structured output: {e}")
                return str(output["content"])
        else:
            # Try to parse the entire output as a JSON string
            try:
                if isinstance(output, str):
                    parsed = json.loads(output)
                    if isinstance(parsed, dict) and "answer" in parsed:
                        result = f"{parsed['answer']}\n\n"
                        
                        if "explanation" in parsed and parsed["explanation"]:
                            result += f"{parsed['explanation']}\n\n"
                            
                        if "citation" in parsed and parsed["citation"]:
                            result += f"Citation: {parsed['citation']}\n\n"
                        
                        return result
            except (json.JSONDecodeError, TypeError):
                pass
                
            return str(output)
    else:
        # Try to parse as JSON string
        try:
            if isinstance(output, str):
                parsed = json.loads(output)
                if isinstance(parsed, dict) and "answer" in parsed:
                    result = f"{parsed['answer']}\n\n"
                    
                    if "explanation" in parsed and parsed["explanation"]:
                        result += f"{parsed['explanation']}\n\n"
                        
                    if "citation" in parsed and parsed["citation"]:
                        result += f"Citation: {parsed['citation']}\n\n"
                    
                    return result
        except (json.JSONDecodeError, TypeError):
            pass
            
        # Regular string output
        return output

if __name__ == "__main__":
    app = MinionsApp()
    app.run(sys.argv)