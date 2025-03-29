#!/usr/bin/env python
"""
AI Group Chat Application

This application creates a group chat environment where multiple AI agents from 
different providers (Google's Gemini, Groq's Llama, etc.) can interact with a human user.

Usage:
    python main.py [--config CONFIG_FILE] [--no-ui]

Options:
    --config CONFIG_FILE    Path to the configuration file (default: config.py)
    --no-ui                 Run in console mode without the terminal UI
    --load HISTORY_FILE     Load a previous chat history

Requirements:
    - Python 3.8+
    - API keys for the AI services you want to use
    - Required packages: pip install -r requirements.txt
"""

import os
import sys
import asyncio
import argparse
import importlib.util
import logging
import json
from typing import Dict, List, Optional, Type
from dotenv import load_dotenv

# Import curses
try:
    import curses
    import curses.ascii
except ImportError:
    if sys.platform == 'win32':
        print("On Windows, you need to install windows-curses:")
        print("pip install windows-curses")
    else:
        print("Curses library not found")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# Import the base classes
from ai_group_chat import (
    Message,
    AIAgent,
    ChatSession,
    GeminiAgent,
    LlamaAgent,
    ClaudeAgent,
    GPTAgent,
)
from terminal_ui import TerminalUI, HumanAgent

# Try to import optional AI client libraries
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

try:
    import openai
    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_chat.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ai_group_chat")

def load_config(config_path: str = "config.py") -> Dict:
    """Load configuration from a Python file."""
    try:
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        # Initialize curses colors
        config_data = {
            "API_KEYS": getattr(config, "API_KEYS", {}),
            "AGENTS": getattr(config, "AGENTS", []),
            "UI_CONFIG": getattr(config, "UI_CONFIG", {}),
            "SESSION_CONFIG": getattr(config, "SESSION_CONFIG", {})
        }
        
        return config_data
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {
            "API_KEYS": {},
            "AGENTS": [],
            "UI_CONFIG": {},
            "SESSION_CONFIG": {}
        }

def create_agent(agent_config: Dict, api_keys: Dict) -> Optional[AIAgent]:
    """Create an AI agent based on the configuration."""
    agent_type = agent_config.get("type", "").lower()
    agent_name = agent_config.get("name", agent_type.capitalize())
    
    if not agent_config.get("enabled", True):
        logger.info(f"Agent {agent_name} is disabled in config")
        return None
    
    # Create the appropriate agent based on type
    if agent_type == "gemini":
        if not GEMINI_AVAILABLE:
            logger.warning(f"Gemini library not installed. Skipping {agent_name}")
            return None
        
        api_key = api_keys.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning(f"GEMINI_API_KEY not found. Skipping {agent_name}")
            return None
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        return GeminiAgent(
            name=agent_name,
            model=agent_config.get("model", "gemini-2.5-pro-exp-03-25"),
            system_prompt=agent_config.get("system_prompt", "")
        )
    
    elif agent_type == "llama":
        if not LLAMA_AVAILABLE:
            logger.warning(f"Groq library not installed. Skipping {agent_name}")
            return None
        
        api_key = api_keys.get("GROQ_API_KEY")
        if not api_key:
            logger.warning(f"GROQ_API_KEY not found. Skipping {agent_name}")
            return None
        
        return LlamaAgent(
            name=agent_name,
            model=agent_config.get("model", "llama-3.3-70b-versatile"),
            system_prompt=agent_config.get("system_prompt", "")
        )
    
    elif agent_type == "anthropic":
        if not CLAUDE_AVAILABLE:
            logger.warning(f"Anthropic library not installed. Skipping {agent_name}")
            return None
        
        api_key = api_keys.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning(f"ANTHROPIC_API_KEY not found. Skipping {agent_name}")
            return None
            
        return ClaudeAgent(
            name=agent_name,
            model=agent_config.get("model", "claude-3-5-sonnet"),
            system_prompt=agent_config.get("system_prompt", "")
        )
    
    elif agent_type == "openai":
        if not GPT_AVAILABLE:
            logger.warning(f"OpenAI library not installed. Skipping {agent_name}")
            return None
        
        api_key = api_keys.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning(f"OPENAI_API_KEY not found. Skipping {agent_name}")
            return None
            
        return GPTAgent(
            name=agent_name,
            model=agent_config.get("model", "gpt-4"),
            system_prompt=agent_config.get("system_prompt", "")
        )
    
    logger.warning(f"Unknown agent type: {agent_type}")
    return None

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Group Chat Application")
    parser.add_argument("--config", default="config.py", help="Path to configuration file")
    parser.add_argument("--no-ui", action="store_true", help="Run in console mode without UI")
    parser.add_argument("--load", help="Load chat history from file")
    args = parser.parse_args()

    try:
        # Initialize curses for color support check
        curses.initscr()
        curses.start_color()
        curses.endwin()
    except Exception as e:
        logger.error(f"Failed to initialize curses: {e}")
        if not args.no_ui:
            print("Terminal does not support required features. Use --no-ui to run in console mode.")
            sys.exit(1)

    # Load configuration
    config = load_config(args.config)
    
    # Create chat session
    if args.load:
        session = ChatSession.from_history(args.load)
    else:
        session = ChatSession()

    # Create and add agents based on configuration
    for agent_config in config["AGENTS"]:
        agent = create_agent(agent_config, config["API_KEYS"])
        if agent:
            session.add_agent(agent)

    # Add human agent
    human = HumanAgent()
    session.add_agent(human)

    if args.no_ui:
        # Run in console mode
        initial_prompt = input("Enter an initial topic or question to start the chat: ")
        await session.run(initial_prompt)
    else:
        # Run with terminal UI
        ui = TerminalUI(session)
        human.set_ui(ui)
        await curses.wrapper(ui.run)

if __name__ == "__main__":
    asyncio.run(main())