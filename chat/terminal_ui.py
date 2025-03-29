import os
import asyncio
import curses
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Import the classes from our main module
from ai_group_chat import Message, AIAgent, GeminiAgent, LlamaAgent, ChatSession

class TerminalUI:
    def __init__(self, session: ChatSession):
        self.session = session
        self.input_buffer = ""
        self.cursor_position = 0
        self.scroll_position = 0
        self.stdscr = None
        self.max_y = 0
        self.max_x = 0
        self.input_height = 3
        self.status_height = 2
        
        # Colors for different agents
        self.colors = {
            "Human": 1,
            "Gemini": 2,
            "Llama": 3,
            "System": 4
        }
    
    def setup_colors(self):
        """Setup color pairs for the UI."""
        curses.start_color()
        curses.use_default_colors()
        
        # Define color pairs for each agent
        curses.init_pair(1, curses.COLOR_WHITE, -1)  # Human: white on default
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # Gemini: green on default
        curses.init_pair(3, curses.COLOR_BLUE, -1)   # Llama: blue on default
        curses.init_pair(4, curses.COLOR_YELLOW, -1) # System: yellow on default
    
    def format_message(self, message: Message) -> str:
        """Format a message for display."""
        timestamp = datetime.fromtimestamp(message.timestamp).strftime("%H:%M:%S")
        return f"[{timestamp}] {message.sender}: {message.content}"
    
    def draw_messages(self):
        """Draw the message history in the chat window."""
        # Calculate chat window dimensions
        chat_height = self.max_y - self.input_height - self.status_height
        chat_width = self.max_x
        
        # Create a chat window
        chat_win = curses.newwin(chat_height, chat_width, 0, 0)
        chat_win.clear()
        
        # Display messages
        y_pos = 0
        messages_to_show = []
        lines_needed = 0
        
        # Calculate how many messages we can show
        for message in reversed(self.session.history):
            formatted = self.format_message(message)
            msg_lines = len([formatted[i:i+chat_width] for i in range(0, len(formatted), chat_width)])
            if lines_needed + msg_lines > chat_height:
                break
            messages_to_show.insert(0, (message, msg_lines))
            lines_needed += msg_lines
        
        # Display the messages
        for message, lines in messages_to_show:
            color = self.colors.get(message.sender, 0)
            formatted = self.format_message(message)
            
            # Split message into lines that fit the window width
            lines_to_print = [formatted[i:i+chat_width] for i in range(0, len(formatted), chat_width)]
            
            for line in lines_to_print:
                chat_win.addstr(y_pos, 0, line, curses.color_pair(color))
                y_pos += 1
        
        chat_win.refresh()
    
    def draw_input_box(self):
        """Draw the input box at the bottom of the screen."""
        input_win = curses.newwin(self.input_height, self.max_x, self.max_y - self.input_height - self.status_height, 0)
        input_win.clear()
        
        # Draw a box around the input area
        input_win.box()
        
        # Show the current input
        input_win.addstr(1, 2, self.input_buffer[:self.max_x-4])
        
        # Position the cursor
        cursor_x = min(2 + self.cursor_position, self.max_x - 3)
        input_win.move(1, cursor_x)
        
        input_win.refresh()
    
    def draw_status_bar(self):
        """Draw the status bar at the bottom of the screen."""
        status_win = curses.newwin(self.status_height, self.max_x, self.max_y - self.status_height, 0)
        status_win.clear()
        
        # Draw a horizontal line to separate from input
        status_win.hline(0, 0, curses.ACS_HLINE, self.max_x)
        
        # Display participants
        participants = f"Participants: {', '.join(self.session.agents.keys())}"
        status_win.addstr(1, 2, participants[:self.max_x-4])
        
        status_win.refresh()
    
    def handle_input(self, key):
        """Handle keyboard input."""
        if key == curses.KEY_ENTER or key == 10 or key == 13:  # Enter key
            if self.input_buffer:
                return self.input_buffer
            return None
        elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace
            if self.cursor_position > 0:
                self.input_buffer = self.input_buffer[:self.cursor_position-1] + self.input_buffer[self.cursor_position:]
                self.cursor_position -= 1
        elif key == curses.KEY_DC:  # Delete key
            if self.cursor_position < len(self.input_buffer):
                self.input_buffer = self.input_buffer[:self.cursor_position] + self.input_buffer[self.cursor_position+1:]
        elif key == curses.KEY_LEFT:  # Left arrow
            self.cursor_position = max(0, self.cursor_position - 1)
        elif key == curses.KEY_RIGHT:  # Right arrow
            self.cursor_position = min(len(self.input_buffer), self.cursor_position + 1)
        elif key == curses.KEY_HOME:  # Home
            self.cursor_position = 0
        elif key == curses.KEY_END:  # End
            self.cursor_position = len(self.input_buffer)
        elif 32 <= key <= 126:  # Printable characters
            char = chr(key)
            self.input_buffer = self.input_buffer[:self.cursor_position] + char + self.input_buffer[self.cursor_position:]
            self.cursor_position += 1
        
        return None
    
    async def get_user_input(self) -> str:
        """Get user input and handle UI updates."""
        self.input_buffer = ""
        self.cursor_position = 0
        
        while True:
            # Update the UI
            self.draw_messages()
            self.draw_input_box()
            self.draw_status_bar()
            
            # Get a key
            try:
                key = self.stdscr.getch()
            except curses.error:
                await asyncio.sleep(0.1)
                continue
            
            # Handle key input
            result = self.handle_input(key)
            if result is not None:
                return result
    
    async def run(self, stdscr):
        """Run the UI loop."""
        self.stdscr = stdscr
        self.stdscr.clear()
        
        # Hide cursor and disable input echo
        curses.curs_set(1)
        curses.noecho()
        self.stdscr.keypad(1)
        
        # Setup colors
        self.setup_colors()
        
        # Get screen dimensions
        self.max_y, self.max_x = self.stdscr.getmaxyx()
        
        # Add a welcome message
        welcome_message = Message(
            sender="System", 
            content="Welcome to the AI Group Chat! Type a message to start the conversation."
        )
        self.session.add_message(welcome_message)
        
        # Main UI loop
        try:
            while True:
                # Get a message from the user
                user_input = await self.get_user_input()
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    # Add a goodbye message
                    goodbye_message = Message(
                        sender="System",
                        content="Thank you for chatting with us! Goodbye."
                    )
                    self.session.add_message(goodbye_message)
                    self.draw_messages()
                    await asyncio.sleep(2)
                    break
                
                # Add the user's message
                human_name = next((name for name, agent in self.session.agents.items() 
                                 if isinstance(agent, HumanAgent)), "Human")
                user_message = Message(sender=human_name, content=user_input)
                self.session.add_message(user_message)
                
                # Let each AI agent respond
                for name, agent in self.session.agents.items():
                    if isinstance(agent, HumanAgent):
                        continue
                    
                    # Update UI to show "thinking" state
                    thinking_message = Message(
                        sender="System",
                        content=f"{name} is thinking..."
                    )
                    self.session.add_message(thinking_message)
                    self.draw_messages()
                    
                    # Get response from the agent
                    try:
                        response = await agent.process_message(self.session.history[-2])  # Get response to the user message
                        # Remove the thinking message
                        self.session.history.pop()
                        # Add the real response
                        self.session.add_message(response)
                    except Exception as e:
                        # Handle errors gracefully
                        error_message = Message(
                            sender="System",
                            content=f"Error getting response from {name}: {str(e)}"
                        )
                        # Remove the thinking message
                        self.session.history.pop()
                        # Add the error message
                        self.session.add_message(error_message)
        
        except KeyboardInterrupt:
            pass
        
        # Save history before exiting
        self.session.save_history("chat_history.json")
        
        # Restore terminal settings
        curses.echo()
        curses.curs_set(1)

class HumanAgent(AIAgent):
    def __init__(self, name: str = "Human"):
        super().__init__(name)
        self.ui = None
    
    def set_ui(self, ui: TerminalUI):
        """Set the UI for this agent."""
        self.ui = ui
    
    async def process_message(self, message: Message) -> Message:
        """Process a message by getting input from the human via the UI."""
        if self.ui:
            user_input = await self.ui.get_user_input()
            return Message(sender=self.name, content=user_input)
        else:
            # Fallback to console input if no UI is set
            print(f"\n{message.sender}: {message.content}")
            user_input = input(f"\n{self.name}: ")
            return Message(sender=self.name, content=user_input)

async def run_terminal_ui():
    # Create a chat session
    session = ChatSession()
    
    # Add agents
    session.add_agent(GeminiAgent())
    session.add_agent(LlamaAgent())
    
    # Add a human agent
    human = HumanAgent()
    session.add_agent(human)
    
    # Create the terminal UI
    ui = TerminalUI(session)
    
    # Set the UI for the human agent
    human.set_ui(ui)
    
    # Run the UI with curses
    await curses.wrapper(ui.run)

if __name__ == "__main__":
    asyncio.run(run_terminal_ui())