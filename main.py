import gi
import os
import ctypes
import sys

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

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

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf

# Import Minions related modules
from minions.minion import Minion
from minions.minions import Minions
from minions.minions_mcp import SyncMinionsMCP, MCPConfigManager

# Import client modules
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.clients.anthropic import AnthropicClient
from minions.clients.together import TogetherClient
from minions.clients.perplexity import PerplexityAIClient
from minions.clients.openrouter import OpenRouterClient
from desktop.clients.together_desktop import TogetherDesktopClient

# Additional imports
import time
from openai import OpenAI
from PIL import Image
import io
from pydantic import BaseModel
import json

MODEL_MAP = {
    "OpenAI": ["text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"],
    "Anthropic": ["anthropic-cassius-001", "anthropic-cassius-002"],
    "Together": ["together-gpt-001", "together-gpt-002"],
    "Perplexity": ["perplexity-gpt-001", "perplexity-gpt-002"],
    "OpenRouter": ["openrouter-gpt-001", "openrouter-gpt-002"]
}

# OpenAI model pricing per 1M tokens
OPENAI_PRICES = {
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
}

PROVIDER_TO_ENV_VAR_KEY = {
    "OpenAI": "OPENAI_API_KEY",
    "OpenRouter": "OPENROUTER_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Together": "TOGETHER_API_KEY",
    "Perplexity": "PERPLEXITY_API_KEY",
}

# For Minions protocol
class JobOutput(BaseModel):
    answer: str | None
    explanation: str | None
    citation: str | None

class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None

class MinionsApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='com.hazyresearch.minions')
        
        # Provider-related settings
        self.providers = ["OpenAI", "Anthropic", "Together", "Perplexity", "OpenRouter"]
        self.current_provider = "OpenAI"
        self.api_key = ""
        
        # Model settings
        self.local_model_name = "ollama:llama2"
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
        
    def do_activate(self):
        print("Activating application...")
        window = MainWindow(application=self)
        print("Window created, showing...")
        window.show_all()  # Explicitly show all widgets
        print("Window shown")
        
    def initialize_clients(self, local_model_name, remote_model_name, provider, protocol,
                          local_temperature, local_max_tokens, remote_temperature, 
                          remote_max_tokens, api_key, num_ctx=4096, mcp_server_name=None):
        """Initialize the local and remote clients for the Minions protocol."""
        print("Initializing clients...")
        
        try:
            # Initialize local client (always Ollama)
            self.local_client = OllamaClient(
                model_name=local_model_name,
                # api_key=api_key,
                temperature=local_temperature,
                max_tokens=int(local_max_tokens),
                num_ctx=num_ctx,
                structured_output_schema=None,
                use_async=False,
            )
            print(f"Local client initialized with model: {local_model_name}")
            
            # Initialize remote client based on provider
            if provider == "OpenAI":
                self.remote_client = OpenAIClient(
                    model=remote_model_name,
                    api_key=api_key,
                    temperature=remote_temperature,
                    max_tokens=remote_max_tokens
                )
            elif provider == "Anthropic":
                self.remote_client = AnthropicClient(
                    model=remote_model_name,
                    api_key=api_key,
                    temperature=remote_temperature,
                    max_tokens=remote_max_tokens
                )
            elif provider == "Together":
                if '--desktop' in sys.argv:
                    self.remote_client = TogetherDesktopClient(api_key=api_key)
                else:
                    self.remote_client = TogetherClient(
                        model=remote_model_name,
                        api_key=api_key,
                        temperature=remote_temperature,
                        max_tokens=remote_max_tokens
                    )
            elif provider == "Perplexity":
                self.remote_client = PerplexityAIClient(
                    model=remote_model_name,
                    api_key=api_key,
                    temperature=remote_temperature,
                    max_tokens=remote_max_tokens
                )
            elif provider == "OpenRouter":
                self.remote_client = OpenRouterClient(
                    model=remote_model_name,
                    api_key=api_key,
                    temperature=remote_temperature,
                    max_tokens=remote_max_tokens
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
            print(f"Remote client initialized with provider: {provider}, model: {remote_model_name}")
            
            # Initialize Minion or Minions based on protocol
            if protocol == "Minion":
                self.minion = Minion(
                    model_client=self.remote_client,
                )
                print("Minion protocol initialized")
            elif protocol == "Minions":
                if mcp_server_name:
                    mcp = SyncMinionsMCP(server_name=mcp_server_name)
                else:
                    mcp = None
                    
                self.minions = Minions(
                    local_client=self.local_client,
                    remote_client=self.remote_client,
                    mcp=mcp,
                )
                print("Minions protocol initialized")
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
                
            return True, "Clients initialized successfully."
        
        except Exception as e:
            error_msg = f"Error initializing clients: {str(e)}"
            print(error_msg)
            return False, error_msg

class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_default_size(1200, 800)
        print("Window initialized with size 1200x800")
        
        # Main paned container
        self.main_paned = Gtk.Paned.new(Gtk.Orientation.HORIZONTAL)
        self.add(self.main_paned)
        print("Added main paned container")
        
        # Protocol settings
        self.protocol = "Minions"  # Default to Minions protocol
        self.protocol_options = ["Minion", "Minions"]
        
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
        self.protocol_combo.set_active(1)  # Default to Minions
        self.protocol_combo.connect("changed", self.on_protocol_changed)
        
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
        self.sidebar.pack_start(local_model_label, False, False, 0)
        self.sidebar.pack_start(self.local_model_entry, False, False, 0)
        self.sidebar.pack_start(temp_box, False, False, 0)
        self.sidebar.pack_start(upload_btn, False, False, 5)
        self.build_document_section()
        print("Sidebar components created and packed")

    def build_main_content(self):
        print("Building main content...")
        self.main_content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.main_paned.add2(self.main_content)
        print("Main content added to paned container")

        # Chat history
        scrolled = Gtk.ScrolledWindow()
        self.chat_history = Gtk.TextView()
        self.chat_history.set_editable(False)
        self.chat_history.set_wrap_mode(Gtk.WrapMode.WORD)
        scrolled.add(self.chat_history)

        # Input area
        input_box = Gtk.Box(spacing=5)
        self.chat_input = Gtk.Entry()
        send_btn = Gtk.Button(label="Send")
        send_btn.connect("clicked", self.on_send_message)

        input_box.pack_start(self.chat_input, True, True, 0)
        input_box.pack_start(send_btn, False, False, 0)

        self.main_content.pack_start(scrolled, True, True, 0)
        self.main_content.pack_start(input_box, False, False, 5)
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

    def on_provider_changed(self, combo):
        self.update_models()

    def update_models(self):
        self.model_combo.remove_all()
        provider = self.provider_combo.get_active_text()
        for model in MODEL_MAP.get(provider, []):
            self.model_combo.append_text(model)
        self.model_combo.set_active(0)

    def on_api_key_changed(self, entry):
        """Update API key when entry changes"""
        self.props.application.api_key = entry.get_text()
        
    def on_protocol_changed(self, combo):
        """Update protocol when selection changes"""
        self.protocol = combo.get_active_text()
        
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
        
        buffer = self.chat_history.get_buffer()
        buffer.insert(buffer.get_end_iter(), f"You: {message}\n")
        self.chat_input.set_text("")
        
        # Execute the appropriate protocol
        self.run_protocol(message)
        
    def run_protocol(self, task):
        """Run the selected protocol with the given task"""
        app = self.props.application
        
        # Show thinking indicator
        buffer = self.chat_history.get_buffer()
        buffer.insert(buffer.get_end_iter(), "AI: Thinking...\n")
        
        # Validate necessary parameters
        if not app.api_key:
            buffer.insert(buffer.get_end_iter(), "Error: API key is required.\n\n")
            return
            
        # Make sure clients are initialized
        if not app.remote_client:
            success, msg = app.initialize_clients(
                app.local_model_name,
                app.remote_model_name,
                app.current_provider,
                self.protocol,
                app.local_temperature,
                app.local_max_tokens,
                app.remote_temperature,
                app.remote_max_tokens,
                app.api_key
            )
            
            if not success:
                buffer.insert(buffer.get_end_iter(), f"Error initializing clients: {msg}\n\n")
                return
        
        # Extract context from uploaded documents
        context = ""
        doc_metadata = {}
        for i, doc in enumerate(app.uploaded_docs):
            context += f"Document {i+1}: {doc['text']}\n\n"
            doc_metadata[f"doc_{i+1}"] = doc['metadata']
        
        # Run the appropriate protocol
        try:
            if self.protocol == "Minion":
                result = app.minion.run(task=task, context=context)
                buffer.insert(buffer.get_end_iter(), f"AI: {result}\n\n")
            elif self.protocol == "Minions":
                def message_callback(role, message, is_final=True):
                    # Only show final messages for now
                    if is_final:
                        prefix = "Remote AI: " if role == "supervisor" else "Local AI: "
                        Gdk.threads_enter()
                        buffer.insert(buffer.get_end_iter(), f"{prefix}{message}\n")
                        Gdk.threads_leave()
                
                # Execute asynchronously to keep UI responsive
                import threading
                def run_minions():
                    try:
                        results = app.minions.run(
                            task=task,
                            context=context,
                            message_callback=message_callback
                        )
                        Gdk.threads_enter()
                        buffer.insert(buffer.get_end_iter(), f"Execution complete.\n\n")
                        Gdk.threads_leave()
                    except Exception as e:
                        Gdk.threads_enter()
                        buffer.insert(buffer.get_end_iter(), f"Error executing protocol: {str(e)}\n\n")
                        Gdk.threads_leave()
                
                threading.Thread(target=run_minions, daemon=True).start()
            else:
                buffer.insert(buffer.get_end_iter(), f"Error: Unsupported protocol '{self.protocol}'\n\n")
        
        except Exception as e:
            buffer.insert(buffer.get_end_iter(), f"Error executing protocol: {str(e)}\n\n")

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
                if extension == ".pdf":
                    text = extract_text_from_pdf(file_bytes)
                elif extension in [".jpg", ".jpeg", ".png"]:
                    text = extract_text_from_image(file_bytes)
                elif extension in [".txt", ".md", ".py", ".js", ".html", ".css", ".json"]:
                    text = file_bytes.decode("utf-8", errors="replace")
                else:
                    # Try to decode as text
                    try:
                        text = file_bytes.decode("utf-8", errors="replace")
                    except:
                        text = f"[Binary file: {file_name}]"
                
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
                    }
                }
                
                self.props.application.uploaded_docs.append(doc_info)
                
                # Update document list
                row = Gtk.ListBoxRow()
                hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
                label = Gtk.Label(label=file_name)
                hbox.pack_start(label, True, True, 0)
                row.add(hbox)
                self.doc_list.add(row)
                self.doc_list.show_all()
                
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

if __name__ == "__main__":
    if '--desktop' in sys.argv:
        client = TogetherDesktopClient(api_key=os.getenv('TOGETHER_API_KEY'))
    else:
        client = TogetherClient(api_key=os.getenv('TOGETHER_API_KEY'))
    
    app = MinionsApp()
    app.run(sys.argv)