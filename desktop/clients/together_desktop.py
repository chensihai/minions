from minions.clients.together import TogetherClient

class TogetherDesktopClient(TogetherClient):
    """Desktop-specific Together client with GTK integration"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add desktop-specific initialization here
