"""
Grant County Indiana History Chat Application
A Streamlit-based chat interface for historical queries about Grant County, Indiana.
"""

import asyncio
import warnings
from typing import Optional, Dict, List, Any
import hashlib

import streamlit as st
from openai.types.responses import FileSearchTool
from pydantic_ai import Agent
from pydantic_ai.models.openai import (
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.providers.openai import OpenAIProvider
from streamlit.logger import get_logger

# Configuration
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic_ai.models.openai")
logger = get_logger(__name__)

class Config:
    """Application configuration management"""
    WELCOME_MESSAGE = """Welcome to the Grant County History Knowledge Base—a resource focused on 
    the people, places, and events that shaped Grant County before 1920. This archive is designed 
    for researchers, educators, and history enthusiasts seeking accurate insights into our 
    county's early years. To begin exploring, start your conversation below."""

    def __init__(self):
        """Initialize configuration from Streamlit secrets"""
        self.api_key: str = st.secrets['API_KEY']
        self.assistant_id: str = st.secrets['assistant_id']
        self.ai_model: str = st.secrets['ai_model']
        self.instruction: str = st.secrets['instructions']
        self.vector_store_id: str = st.secrets['vector_store'][0]
        # Add password hash to config - replace 'your_password_here' with actual password
        self.password_hash: str = hashlib.sha256(st.secrets['password'].encode()).hexdigest()

def check_password() -> bool:
    """Implement password checking"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Create a form container that can be removed
    login_container = st.empty()
    
    with login_container.container():
        st.subheader("Please Login")
        st.write("During this testing period, we are restricting access to testers with a password.")
        password = st.text_input("Please enter password:", type="password")
        if password:
            if hashlib.sha256(password.encode()).hexdigest() == config.password_hash:
                st.session_state.authenticated = True
                # Remove the login form

                login_container.empty()

                return True
            else:
                st.error("Incorrect password. Please try again.")
                return False
    return False

class ChatAgent:
    """Handles AI chat agent initialization and response streaming"""

    def __init__(self, config: Config):
        """Initialize the chat agent with configuration"""
        self.config = config
        self.model_settings = self._create_model_settings()
        self.model = self._create_model()
        self.agent = self._create_agent()

    def _create_model_settings(self) -> OpenAIResponsesModelSettings:
        """Create and return model settings"""
        return OpenAIResponsesModelSettings(
            openai_builtin_tools=[
                FileSearchTool(
                    type='file_search',
                    vector_store_ids=[self.config.vector_store_id],
                    max_num_results=20,
                )
            ],
        )

    def _create_model(self) -> OpenAIResponsesModel:
        """Create and return OpenAI model"""
        return OpenAIResponsesModel(
            model_name=self.config.ai_model,
            provider=OpenAIProvider(api_key=self.config.api_key)
        )

    def _create_agent(self) -> Agent:
        """Create and return AI agent"""
        return Agent(
            model=self.model,
            model_settings=self.model_settings,
            instructions=self.config.instruction,
        )

    async def stream_response(self, user_input: str) -> Optional[str]:
        """
        Stream the agent's response to a user input

        Args:
            user_input: The user's question or prompt

        Returns:
            The final response text or None if an error occurred
        """
        message_placeholder = st.empty()
        last_message = None

        try:
            async with self.agent.run_stream(user_input) as result:
                async for message in result.stream_text():
                    if message := message.strip():
                        message_placeholder.markdown(message)
                        last_message = message
                        await asyncio.sleep(0.01)

            return last_message
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            return f"An error occurred: {str(e)}"

def run_async(coro: Any) -> Any:
    """
    Utility function to run asynchronous code

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

def init_session_state() -> None:
    """Initialize or reset the session state"""
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = config.ai_model
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history() -> None:
    """Display the chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main() -> None:
    """Main application function"""
    if not check_password():
        return

    st.title("Grant County Indiana History")
    st.write(Config.WELCOME_MESSAGE)

    init_session_state()
    display_chat_history()

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = run_async(chat_agent.stream_response(prompt))
                if response:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    logger.info(
                        f"\n<========\nprompt: {prompt}\nresult: {response}\n========>\n"
                    )

if __name__ == "__main__":
    config = Config()
    chat_agent = ChatAgent(config)
    main()
