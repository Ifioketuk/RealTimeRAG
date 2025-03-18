import os
import boto3
import json
import streamlit as st

# App title
st.set_page_config(page_title="Yharn App", layout="wide")


import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from dotenv import load_dotenv
from pinecone import Pinecone
import requests
import time
import threading

# Import our custom audio recorder component
from audiorecorder import audio_recorder

load_dotenv()

# Access environment variables
aws_region = os.getenv("AWS_REGION")
modelId = os.getenv("MODEL_ID")
emb_modelId = os.getenv("EMB_MODEL_ID")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
fastapi_url = os.getenv("FASTAPI_URL", "https://realtimerag-1.onrender.com")

# Store FastAPI URL in session state for the audio recorder component
if 'fastapi_url' not in st.session_state:
    st.session_state.fastapi_url = fastapi_url

# Initialize transcription lock if not exists
if 'transcription_lock' not in st.session_state:
    st.session_state.transcription_lock = threading.Lock()

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
    return response.status_code == 200

def stop_transcription():
    response = requests.post(f"{fastapi_url}/stop_transcription")
    return response.status_code == 200

def get_transcription():
    response = requests.get(f"{fastapi_url}/get_transcription")
    if response.status_code == 200:
        return response.json().get("transcription", "")
    return ""

def transcription_loop():
    while st.session_state.transcribing:
        text = get_transcription()
        if text:
            with st.session_state.transcription_lock:
                st.session_state.transcription_text += text + " "
        time.sleep(0.5)  # Check every half second

def create_embedding(text):
    try:
        response = bedrock.invoke_model(
            modelId=emb_modelId,
            body=json.dumps({
                "inputText": text
            })
        )
        response_body = json.loads(response.get('body').read())
        return response_body.get('embedding')
    except Exception as e:
        st.error(f"Error creating embedding: {str(e)}")
        return [0.1] * 1536  # Fallback placeholder

def get_answer_from_event(question):
    try:
        # Create embedding for the question
        question_embedding = create_embedding(question)
        
        # Query Pinecone for relevant context
        query_response = index.query(
            vector=question_embedding,
            top_k=3,
            include_metadata=True
        )
        
        # Extract context from results
        context = ""
        for match in query_response['matches']:
            if 'text' in match['metadata']:
                context += match['metadata']['text'] + "\n"
        
        # Generate answer using Bedrock
        prompt = f"""
        Answer the following question based on the provided context:
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        # Call Bedrock model
        response = bedrock.invoke_model(
            modelId=modelId,
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 500,
                "temperature": 0.7,
                "top_p": 0.9,
            })
        )
        
        response_body = json.loads(response.get('body').read())
        return response_body.get('completion', "I couldn't find an answer to that question.")
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"



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
    # YK user - Transcription interface
    elif st.session_state["name"] == 'yk':
            st.title("Welcome to Yharn Transcribe 🎙️")
            st.sidebar.title(f"Welcome,Yinka 😝 ")
            
            # Initialize transcription state
            if "transcription_text" not in st.session_state:
                st.session_state.transcription_text = ""
            
            if "transcribing" not in st.session_state:
                st.session_state.transcribing = False
            
            # Audio recorder component
            st.write("Use the controls below to record audio:")
            audio_recorder()
            
            # Manual transcription controls
            st.write("Or control transcription manually:")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Start Transcription"):
                    if start_transcription():
                        st.session_state.transcription_text = ""
                        st.session_state.transcribing = True
                        # Start transcription in a background thread
                        thread = threading.Thread(target=transcription_loop, daemon=True)
                        thread.start()
                        st.success("Transcription started!")
            
            with col2:
                if st.button("Stop Transcription"):
                    if stop_transcription():
                        st.session_state.transcribing = False
                        st.success("Transcription stopped!")
            
            # Live transcription display
            st.subheader("Live Transcription")
            transcription_container = st.empty()
            
            # Use a placeholder to update the transcription text
            with transcription_container.container():
                st.text_area(
                    "Transcribed Text",
                    value=st.session_state.transcription_text,
                    height=300,
                    key="transcription_display"
                )
            
            # Display transcription status
            status_text = "🔴 Not Recording" if not st.session_state.transcribing else "🟢 Recording"
            st.sidebar.markdown(f"**Status:** {status_text}")
                
            # Logout button
            with st.sidebar:
                if authenticator.logout('Logout', 'sidebar'):
                    st.session_state.clear()
                    st.experimental_rerun()
    
    # Other authenticated users
    

# Authentication failed
    else:
        st.write(f"Welcome {st.session_state['name']}!")
        authenticator.logout('Logout', 'main')

elif st.session_state.get("authentication_status") is False:
    st.error('Username/password is incorrect')
elif st.session_state.get("authentication_status") is None:
    st.warning('Please enter your username and password')
