# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys Configuration
API_KEYS = {
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
}

# Agent Configuration
AGENTS = [
    {
        "name": "Gemini",
        "type": "gemini",
        "model": "gemini-2.5-pro-exp-03-25",
        "enabled": True,
        "system_prompt": "You are Gemini, an AI assistant in a group chat with other AI models and a human. Keep your responses informative but concise. You excel at providing factual information and logical analysis."
    },
    {
        "name": "Llama",
        "type": "llama",
        "model": "llama-3.3-70b-versatile",
        "enabled": True,
        "system_prompt": "You are Llama, an AI assistant in a group chat with other AI models and a human. You're known for your creative and out-of-the-box thinking. Keep responses concise and focus on generating novel ideas."
    },
    {
        "name": "Claude",
        "type": "anthropic",
        "model": "claude-3-5-sonnet",
        "enabled": False,  # Set to True when you want to add Claude
        "system_prompt": "You are Claude, an AI assistant in a group chat with other AI models and a human. You excel at nuanced reasoning and thoughtful analysis. Provide insights that complement the other AI assistants."
    },
    {
        "name": "GPT",
        "type": "openai",
        "model": "gpt-4o",
        "enabled": False,  # Set to True when you want to add GPT
        "system_prompt": "You are GPT, an AI assistant in a group chat with other AI models and a human. You're versatile and adaptive, known for your wide-ranging capabilities. Focus on providing well-balanced and helpful responses."
    }
]

# UI Configuration
UI_CONFIG = {
    "colors": {
        "Human": (7, -1),     # White on default background
        "Gemini": (2, -1),    # Green on default background
        "Llama": (4, -1),     # Blue on default background
        "Claude": (5, -1),    # Magenta on default background
        "GPT": (6, -1),       # Cyan on default background
        "System": (3, -1)     # Yellow on default background
    },
    "refresh_rate": 0.1,      # Seconds between UI refreshes
    "max_history": 100,       # Maximum number of messages to keep in history
    "max_context": 10         # Maximum number of messages to include in context for AI
}

# Chat Session Configuration
SESSION_CONFIG = {
    "history_file": "chat_history.json",
    "log_level": "INFO"  # DEBUG, INFO, WARNING, ERROR
}