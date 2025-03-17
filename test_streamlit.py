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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from main import basic_transcribe  # Import real-time transcription function

load_dotenv()

# Access environment variables
aws_region = os.getenv("AWS_REGION")
modelId = os.getenv("MODEL_ID")
emb_modelId = os.getenv("EMB_MODEL_ID")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

# Create session and clients
bedrock = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=pinecone_api_key)
    return pc.Index(index_name)

# Initialize the Pinecone index
index = init_pinecone()

prompt_template = """ 
You are an AI assistant with access to knowledge about any event or conversation. You respond to the user question as if you have the event or conversation in your knowledge base.

Your Responsibilities: 
1. Answer questions about the event by using relevant information retrieved. 
2. Your responses should be conversational, clear, and use simple grammar to ensure easy understanding. 
3. If specific information is not in the transcript, let the user know politely.
4. Be affirming with your responses. For example:
    Never use "seems" in your responses like: "It seems like the last point made was about funding."
    Instead, say: "The last point made was about funding."

<context>
{context}
</context>

Question: {question}

Helpful Answer:
"""

def get_answer_from_event(query):
    input_data = {
        "inputText": query,
        "dimensions": 1024,
        "normalize": True
    }

    body = json.dumps(input_data).encode('utf-8')
    response = bedrock.invoke_model(
        modelId=emb_modelId,
        contentType="application/json",
        accept="*/*",
        body=body
    )

    response_body = response['body'].read()
    response_json = json.loads(response_body)
    query_embedding = response_json['embedding']

    result = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    context = [f"Score: {match['score']}, Metadata: {match['metadata']}" for match in result['matches']]
    context_string = "\n".join(context)
    
    message_list = [{"role": "user", "content": [{"text": query}]}]
    response = bedrock.converse(
        modelId=modelId,
        messages=message_list,
        system=[
            {"text": prompt_template.format(context=context_string, question=query)},
        ],
        inferenceConfig={"maxTokens": 2000, "temperature": 1},
    )
    
    response_message = response['output']['message']['content'][0]['text']
    return response_message

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

        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        webrtc_ctx = webrtc_streamer(
            key="audio-only",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": False, "audio": True},
        )

        if st.button("Start Transcription"):
            st.session_state.transcription_text = ""  # Reset text
            st.session_state.transcription_running = True
            asyncio.create_task(basic_transcribe())

        if st.button("Stop Transcription"):
            st.session_state.transcription_running = False

        st.text_area("Live Transcription", value=st.session_state.transcription_text, height=200)

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
