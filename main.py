import gi
import os
import ctypes
import sys

# Windows-specific configuration
if os.name == 'nt':
    # Add GTK runtime to PATH
    os.environ["PATH"] = r"C:\Program Files\GTK3-Runtime Win64\bin;" + os.environ.get("PATH", "")
    
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

MODEL_MAP = {
    "OpenAI": ["text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"],
    "Anthropic": ["anthropic-cassius-001", "anthropic-cassius-002"],
    "Together": ["together-gpt-001", "together-gpt-002"],
    "Perplexity": ["perplexity-gpt-001", "perplexity-gpt-002"],
    "OpenRouter": ["openrouter-gpt-001", "openrouter-gpt-002"]
}

class MinionsApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='com.hazyresearch.minions')
        self.providers = ["OpenAI", "Anthropic", "Together", "Perplexity", "OpenRouter"]
        self.current_provider = "OpenAI"
        self.api_key = ""
        
    def do_activate(self):
        print("Activating application...")
        window = MainWindow(application=self)
        print("Window created, showing...")
        window.show_all()  # Explicitly show all widgets
        print("Window shown")

class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_default_size(1200, 800)
        print("Window initialized with size 1200x800")
        
        # Main paned container
        self.main_paned = Gtk.Paned.new(Gtk.Orientation.HORIZONTAL)
        self.add(self.main_paned)
        print("Added main paned container")
        
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

        # Model selection
        model_label = Gtk.Label(label="Model:")
        self.model_combo = Gtk.ComboBoxText()
        self.update_models()

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

    def on_upload_clicked(self, widget):
        dialog = Gtk.FileChooserNative(
            title="Choose Documents",
            transient_for=self,
            action=Gtk.FileChooserAction.OPEN
        )
        dialog.connect("response", self.on_file_selected)
        dialog.show()

    def on_send_message(self, widget):
        message = self.chat_input.get_text()
        if not message:
            return
        
        buffer = self.chat_history.get_buffer()
        buffer.insert(buffer.get_end_iter(), f"You: {message}\n")
        self.chat_input.set_text("")
        
        # TODO: Implement AI response
        buffer.insert(buffer.get_end_iter(), f"AI: Simulated response to: {message}\n\n")

    def on_file_selected(self, dialog, response):
        if response == Gtk.ResponseType.ACCEPT:
            file_path = dialog.get_filename()
            # TODO: Implement document upload logic

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
    # Debug settings
    print(f"Python: {sys.version}")
    print(f"GTK+: {Gtk.get_major_version()}.{Gtk.get_minor_version()}.{Gtk.get_micro_version()}")
    
    # Create app
    app = MinionsApp()
    
    # Run with tracing
    print("Starting app...")
    app.run(None)
    print("App exited")