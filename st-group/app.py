import streamlit as st
import os
import time
from google import genai
from google.genai import types
from groq import Groq

# Set page configuration
st.set_page_config(
    page_title="Multi-Agent Group Chat",
    page_icon="💬",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_typing" not in st.session_state:
    st.session_state.agent_typing = None

# Define the agents
AGENTS = {
    "You": {"avatar": "👤", "color": "#1E88E5"},
    "Gemini": {"avatar": "🧠", "color": "#43A047"},
    "Llama": {"avatar": "🦙", "color": "#FB8C00"}
}

# Add CSS for chat styling
st.markdown("""
<style>
.chat-container {
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.agent-name {
    font-weight: bold;
    margin-bottom: 5px;
}
.typing-indicator {
    display: inline-block;
    width: 20px;
    text-align: center;
}
.typing-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #888;
    margin: 0 2px;
    animation: typing 1.4s infinite both;
}
.typing-dot:nth-child(2) {
    animation-delay: 0.2s;
}
.typing-dot:nth-child(3) {
    animation-delay: 0.4s;
}
@keyframes typing {
    0% { opacity: 0.2; }
    20% { opacity: 1; }
    100% { opacity: 0.2; }
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("Multi-Agent Group Chat")
st.markdown("Chat with multiple AI agents and share ideas together!")

# Sidebar for configuration
st.sidebar.header("Configuration")

# API key inputs
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

# Context area
st.sidebar.header("Chat Context")
context = st.sidebar.text_area("Context for the AI agents", 
    "You are part of a group chat discussing interesting tech innovations. Be friendly and concise.", 
    height=100)

# Helper functions
def get_gemini_response(prompt):
    """Get response from Gemini model"""
    try:
        client = genai.Client(api_key=gemini_api_key)
        model = "gemini-2.5-pro-exp-03-25"
        
        full_prompt = f"{context}\n\nYou're the Gemini AI in a group chat. Previous messages: {format_history_for_context()}\n\nUser message: {prompt}"
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=full_prompt)],
            ),
        ]
        
        tools = [types.Tool(google_search=types.GoogleSearch())]
        
        generate_content_config = types.GenerateContentConfig(
            tools=tools,
            response_mime_type="text/plain",
        )
        
        response_text = ""
        st.session_state.agent_typing = "Gemini"
        
        # Simulate streaming for UI
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text if chunk.text else ""
            time.sleep(0.05)  # Slight delay for visual effect
            rerender_chat()
            
        st.session_state.agent_typing = None
        return response_text
    except Exception as e:
        st.session_state.agent_typing = None
        return f"Error with Gemini: {str(e)}"

def get_llama_response(prompt):
    """Get response from Llama model via Groq"""
    try:
        client = Groq(api_key=groq_api_key)
        
        # Format chat history for Llama
        messages = [{"role": "system", "content": context}]
        
        # Add chat history
        for msg in st.session_state.messages:
            role = "assistant" if msg["agent"] != "You" else "user"
            messages.append({"role": role, "content": f"{msg['agent']}: {msg['content']}"})
        
        # Add the current message
        messages.append({"role": "user", "content": prompt})
        
        response_text = ""
        st.session_state.agent_typing = "Llama"
        
        # Get streaming response
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=True,
        )
        
        for chunk in completion:
            chunk_text = chunk.choices[0].delta.content or ""
            response_text += chunk_text
            time.sleep(0.03)  # Slight delay for visual effect
            rerender_chat()
            
        st.session_state.agent_typing = None
        return response_text
    except Exception as e:
        st.session_state.agent_typing = None
        return f"Error with Llama: {str(e)}"

def format_history_for_context():
    """Format chat history for context"""
    history_text = ""
    for msg in st.session_state.messages[-10:]:  # Last 10 messages
        history_text += f"{msg['agent']}: {msg['content']}\n"
    return history_text

def add_message(agent, content):
    """Add a message to chat history"""
    st.session_state.messages.append({"agent": agent, "content": content})

def rerender_chat():
    """Force rerender of chat"""
    chat_placeholder.empty()
    display_chat()

# Display chat messages
def display_chat():
    for message in st.session_state.messages:
        agent = message["agent"]
        agent_info = AGENTS[agent]
        
        # Create message container with agent-specific styling
        with st.container():
            st.markdown(
                f"""
                <div class="chat-container" style="background-color: {agent_info['color']}25;">
                    <div class="agent-name">{agent_info['avatar']} {agent}</div>
                    <div>{message['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Show typing indicator if an agent is typing
    if st.session_state.agent_typing:
        agent = st.session_state.agent_typing
        agent_info = AGENTS[agent]
        
        with st.container():
            st.markdown(
                f"""
                <div class="chat-container" style="background-color: {agent_info['color']}25;">
                    <div class="agent-name">{agent_info['avatar']} {agent}</div>
                    <div class="typing-indicator">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# Create a placeholder for the chat messages
chat_placeholder = st.empty()
display_chat()

# User input
user_input = st.text_input("Your message:", key="user_input")

# Agent selection for response
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Send to All Agents"):
        if user_input and gemini_api_key and groq_api_key:
            # Add user message
            add_message("You", user_input)
            rerender_chat()
            
            # Get Gemini response
            gemini_response = get_gemini_response(user_input)
            add_message("Gemini", gemini_response)
            rerender_chat()
            
            # Get Llama response
            llama_response = get_llama_response(user_input)
            add_message("Llama", llama_response)
            rerender_chat()
            
            # Clear the input
            st.session_state.user_input = ""

with col2:
    if st.button("Send to Gemini"):
        if user_input and gemini_api_key:
            # Add user message
            add_message("You", user_input)
            rerender_chat()
            
            # Get Gemini response
            gemini_response = get_gemini_response(user_input)
            add_message("Gemini", gemini_response)
            rerender_chat()
            
            # Clear the input
            st.session_state.user_input = ""

with col3:
    if st.button("Send to Llama"):
        if user_input and groq_api_key:
            # Add user message
            add_message("You", user_input)
            rerender_chat()
            
            # Get Llama response
            llama_response = get_llama_response(user_input)
            add_message("Llama", llama_response)
            rerender_chat()
            
            # Clear the input
            st.session_state.user_input = ""

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    rerender_chat()

# Display API key warnings if not provided
if not gemini_api_key:
    st.sidebar.warning("Please enter your Gemini API Key to enable Gemini responses.")
    
if not groq_api_key:
    st.sidebar.warning("Please enter your Groq API Key to enable Llama responses.")

# Add info about adding more agents
st.sidebar.header("Adding More Agents")
st.sidebar.info("""
To add more agents to this chat:
1. Create API access for other LLM providers
2. Add their client code similar to existing agents
3. Update the AGENTS dictionary with new agent details
4. Create response functions for each new agent
5. Add new buttons for agent-specific interactions
""")