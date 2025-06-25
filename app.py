import streamlit as st
from openai.types.responses import FileSearchTool

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

API_KEY = st.secrets['API_KEY']  # Replace with your actual API key
assistant_id = st.secrets['assistant_id']
AI_MODEL = st.secrets['ai_model']

instruction = st.secrets['instructions']

st.title("Grant County Indiana History")
st.write(
    "Welcome to the Grant County History Knowledge Base—a resource focused on the people, places, and events that shaped Grant County before 1920. This archive is designed for researchers, educators, and history enthusiasts seeking accurate insights into our county’s early years. To begin exploring, start your conversation below.")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = AI_MODEL

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Who was Moses Bradford?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing...", show_time=True):
            model_settings = OpenAIResponsesModelSettings(
                openai_builtin_tools=[FileSearchTool(
                    type='file_search',
                    vector_store_ids=['vs_6859f79bc33c8191ae9a961d6b98da9a'],
                    max_num_results=20,
                )],
            )
            model = OpenAIResponsesModel(model_name=AI_MODEL, provider=OpenAIProvider(api_key=API_KEY))
            agent = Agent(
                model=model,
                model_settings=model_settings,
                instructions=instruction,

            )
            result = agent.run_sync(prompt)
            print(result.output)

            # print(json.dumps(response.output[1]))

            st.markdown(result.output)
            st.session_state.messages.append({"role": "assistant", "content": result.output})
