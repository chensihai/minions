from setuptools import setup, find_packages

setup(
    name="minions-desktop",
    version="0.1.0",
    description="Desktop application for the Minions project",
    author="Sabri, Avanika, and Dan",
    packages=find_packages(),
    install_requires=[
        "pygobject==3.50.0",
        "requests==2.31.0",
        "python-dotenv==1.0.1",
        "pydantic==2.10.6",
        "openai",
        "anthropic",
        "together",
        "tiktoken",
        "Pillow",
        "st-theme",
        "mcp",
        "spacy",
        "ollama"
    ],
    entry_points={
        "console_scripts": [
            "minions-desktop=main:main",
        ],
    },
)