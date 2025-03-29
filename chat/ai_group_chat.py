import os
import asyncio
import json
import base64
from typing import List, Dict, Optional
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# For Gemini
import google.generativeai as genai
from google.generativeai import types as genai_types

# For Llama (via Groq)
from groq import Groq

# Optional: For Claude (if you decide to add it)
# import anthropic

class Message:
    def __init__(self, sender: str, content: str, timestamp: Optional[float] = None):
        self.sender = sender
        self.content = content
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        return cls(
            sender=data["sender"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time())
        )

class AIAgent:
    def __init__(self, name: str):
        self.name = name
        self.context: List[Message] = []
    
    async def process_message(self, message: Message) -> Message:
        """Process a message and return a response."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def update_context(self, message: Message):
        """Add a message to the agent's context."""
        self.context.append(message)

class GeminiAgent(AIAgent):
    def __init__(self, name: str = "Gemini", model: str = "gemini-2.5-pro-exp-03-25", system_prompt: str = ""):
        super().__init__(name)
        # Initialize the Gemini client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.model = model
        self.system_prompt = system_prompt or "You are Gemini, an AI assistant in a group chat with other AI models and a human. Keep your responses concise and contribute meaningfully to the conversation."
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(
            model_name=self.model,
            generation_config={"temperature": 0.7}
        )
    
    async def process_message(self, message: Message) -> Message:
        # Prepare the conversation history for Gemini
        contents = []
        
        # Add system prompt
        contents.append({"role": "system", "parts": [{"text": self.system_prompt}]})
        
        # Add context messages
        for ctx_msg in self.context[-10:]:  # Limit context to last 10 messages
            role = "user" if ctx_msg.sender != self.name else "model"
            contents.append({"role": role, "parts": [{"text": f"{ctx_msg.sender}: {ctx_msg.content}"}]})
        
        # Add the new message
        contents.append({"role": "user", "parts": [{"text": f"{message.sender}: {message.content}"}]})
        
        # Get response from Gemini
        response = await asyncio.to_thread(
            self.client.generate_content,
            contents
        )
        
        return Message(sender=self.name, content=response.text)

class LlamaAgent(AIAgent):
    def __init__(self, name: str = "Llama", model: str = "llama-3.3-70b-versatile", system_prompt: str = ""):
        super().__init__(name)
        # Initialize the Groq client for Llama
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt or "You are Llama, an AI assistant in a group chat with other AI models and a human. Keep your responses concise and contribute meaningfully to the conversation."
    
    async def process_message(self, message: Message) -> Message:
        # Prepare the conversation history for Llama via Groq
        messages = []
        
        # Add system prompt
        messages.append({
            "role": "system", 
            "content": self.system_prompt
        })
        
        # Add context messages
        for ctx_msg in self.context[-10:]:  # Limit context to last 10 messages
            role = "user" if ctx_msg.sender != self.name else "assistant"
            messages.append({"role": role, "content": f"{ctx_msg.sender}: {ctx_msg.content}"})
        
        # Add the new message
        messages.append({"role": "user", "content": f"{message.sender}: {message.content}"})
        
        # Get response from Llama via Groq
        completion = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        return Message(sender=self.name, content=completion.choices[0].message.content)

class ClaudeAgent(AIAgent):
    def __init__(self, name: str = "Claude", model: str = "claude-3-5-sonnet", system_prompt: str = ""):
        super().__init__(name)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Client(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt

    async def process_message(self, message: Message) -> Message:
        messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add context messages
        for ctx_msg in self.context[-10:]:
            role = "user" if ctx_msg.sender != self.name else "assistant"
            messages.append({"role": role, "content": f"{ctx_msg.sender}: {ctx_msg.content}"})
        
        # Add the new message
        messages.append({"role": "user", "content": f"{message.sender}: {message.content}"})
        
        # Get response from Claude
        completion = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.7
        )
        
        return Message(sender=self.name, content=completion.content)

class GPTAgent(AIAgent):
    def __init__(self, name: str = "GPT", model: str = "gpt-4", system_prompt: str = ""):
        super().__init__(name)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt

    async def process_message(self, message: Message) -> Message:
        messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add context messages
        for ctx_msg in self.context[-10:]:
            role = "user" if ctx_msg.sender != self.name else "assistant"
            messages.append({"role": role, "content": f"{ctx_msg.sender}: {ctx_msg.content}"})
        
        # Add the new message
        messages.append({"role": "user", "content": f"{message.sender}: {message.content}"})
        
        # Get response from GPT
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.7
        )
        
        return Message(sender=self.name, content=completion.choices[0].message.content)

class HumanAgent(AIAgent):
    def __init__(self, name: str = "Human"):
        super().__init__(name)
    
    async def process_message(self, message: Message) -> Message:
        # For a human agent, we just ask for input
        print(f"\n{message.sender}: {message.content}")
        user_input = input(f"\n{self.name} (you): ")
        return Message(sender=self.name, content=user_input)

class ChatSession:
    def __init__(self):
        self.agents: Dict[str, AIAgent] = {}
        self.history: List[Message] = []
    
    def add_agent(self, agent: AIAgent):
        """Add an agent to the chat session."""
        self.agents[agent.name] = agent
    
    def add_message(self, message: Message):
        """Add a message to the chat history and update agent contexts."""
        self.history.append(message)
        # Update context for all agents
        for agent in self.agents.values():
            agent.update_context(message)
    
    async def run(self, initial_prompt: str):
        """Run the chat session."""
        # Add the initial message from the human
        human_name = next((name for name, agent in self.agents.items() if isinstance(agent, HumanAgent)), "Human")
        initial_message = Message(sender=human_name, content=initial_prompt)
        self.add_message(initial_message)
        
        print(f"\nWelcome to the AI Group Chat!")
        print(f"Participants: {', '.join(self.agents.keys())}")
        print(f"\n{initial_message.sender}: {initial_message.content}")
        
        # Main chat loop
        try:
            while True:
                # Each agent (except the human) responds to the last message
                for name, agent in self.agents.items():
                    # Skip the human agent for automatic responses
                    if isinstance(agent, HumanAgent):
                        continue
                    
                    # Get response from the agent
                    response = await agent.process_message(self.history[-1])
                    self.add_message(response)
                    
                    # Print the response
                    print(f"\n{response.sender}: {response.content}")
                
                # Get input from the human
                human_agent = next((agent for agent in self.agents.values() if isinstance(agent, HumanAgent)), None)
                if human_agent:
                    human_response = await human_agent.process_message(self.history[-1])
                    self.add_message(human_response)
                    
                    # Check if the user wants to exit
                    if human_response.content.lower() in ["exit", "quit", "bye"]:
                        print("\nThank you for chatting with us! Goodbye.")
                        break
        
        except KeyboardInterrupt:
            print("\n\nChat session ended by user.")
            
        # Save chat history
        self.save_history("chat_history.json")
    
    def save_history(self, filename: str):
        """Save the chat history to a file."""
        with open(filename, "w") as f:
            json.dump([msg.to_dict() for msg in self.history], f, indent=2)
        print(f"\nChat history saved to {filename}")
    
    @classmethod
    def from_history(cls, filename: str) -> 'ChatSession':
        """Load a chat session from a history file."""
        session = cls()
        try:
            with open(filename, "r") as f:
                history_data = json.load(f)
                for msg_data in history_data:
                    session.history.append(Message.from_dict(msg_data))
        except FileNotFoundError:
            pass
        return session

async def main():
    # Create a chat session
    session = ChatSession()
    
    # Add agents
    session.add_agent(GeminiAgent())
    session.add_agent(LlamaAgent())
    session.add_agent(HumanAgent())
    
    # Start the chat
    initial_prompt = input("Enter an initial topic or question to start the chat: ")
    await session.run(initial_prompt)

if __name__ == "__main__":
    asyncio.run(main())