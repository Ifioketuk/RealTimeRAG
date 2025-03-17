import os
import boto3
import asyncio
import json
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from dotenv import load_dotenv
from pinecone import Pinecone
import requests
import time

load_dotenv()

# Access environment variables
aws_region = os.getenv("AWS_REGION")
modelId = os.getenv("MODEL_ID")
emb_modelId = os.getenv("EMB_MODEL_ID")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
fastapi_url = os.getenv("FASTAPI_URL","https://realtimerag-1.onrender.com")  # URL for FastAPI backend

# Create session and clients
bedrock = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(index_name)

# Initialize the Pinecone index
index = init_pinecone()

def start_transcription():
    response = requests.post(f"{fastapi_url}/start_transcription")
    if response.status_code == 200:
        st.session_state.transcribing = True

def stop_transcription():
    response = requests.post(f"{fastapi_url}/stop_transcription")
    if response.status_code == 200:
        st.session_state.transcribing = False

def get_transcription():
    response = requests.get(f"{fastapi_url}/get_transcription")
    if response.status_code == 200:
        return response.json().get("transcription", "")
    return ""

def transcription_loop():
    while st.session_state.transcribing:
        text = get_transcription()
        if text:
            st.session_state.transcription_text += text + " "
        time.sleep(1)

# Load configuration
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

if 'authenticator' not in st.session_state:
    st.session_state['authenticator'] = authenticator

authenticator.login('main')

# Check authentication status
if st.session_state.get("authentication_status"):
    if st.session_state["name"] == 'oracle':
        st.title("Yharn Chat 🤖")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if len(st.session_state.messages) == 0:
            assistant_message = "Hello! How can I assist you with the event today?"
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("Generating response..."):
                assistant_response = get_answer_from_event(user_input)

            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        st.markdown("<br>", unsafe_allow_html=True)

        with st.sidebar:
            if authenticator.logout('Logout', 'main'):
                st.session_state.clear()
                st.write("You have logged out successfully!")
                st.stop()

    elif st.session_state["name"] == 'yk':
        st.title("Welcome to Yharn Transcribe 🎙️")
        st.write("Click 'Start' to begin real-time transcription.")

        if "transcription_text" not in st.session_state:
            st.session_state.transcription_text = ""

        if "transcribing" not in st.session_state:
            st.session_state.transcribing = False

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Transcription"):
                start_transcription()
                st.session_state.transcription_text = ""
                st.session_state.transcribing = True
                asyncio.create_task(transcription_loop())
        
        with col2:
            if st.button("Stop Transcription"):
                stop_transcription()

        st.text_area("Live Transcription", value=st.session_state.transcription_text, height=300)

        with st.sidebar:
            if authenticator.logout('Logout', 'main'):
                st.session_state.clear()
                st.write("You have logged out successfully!")
                st.stop()

    else:
        st.write(f"Welcome {st.session_state['name']}!")
        authenticator.logout('Logout', 'main')

elif st.session_state.get("authentication_status") is False:
    st.error('Username/password is incorrect')
elif st.session_state.get("authentication_status") is None:
    st.warning('Please enter your username and password')
