import streamlit as st
import streamlit.components.v1 as components

def audio_recorder():
    """
    Embeds an HTML/JavaScript component for audio recording.
    The component sends audio data directly to the FastAPI backend.
    """
    # HTML and JavaScript for audio recording
    audio_recorder_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Audio Recorder</title>
        <style>
            .recorder-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-bottom: 10px;
            }
            button {
                padding: 10px 20px;
                margin: 5px;
                border: none;
                border-radius: 5px;
                background-color: #4CAF50;
                color: white;
                cursor: pointer;
            }
            button:disabled {
                background-color: #cccccc;
            }
            .status {
                margin-top: 10px;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="recorder-container">
            <div>
                <button id="startButton">Start Recording</button>
                <button id="stopButton" disabled>Stop Recording</button>
            </div>
            <div class="status" id="status">Ready to record</div>
        </div>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const statusElement = document.getElementById('status');
            const apiUrl = 'FASTAPI_URL_PLACEHOLDER/send_audio';

            startButton.addEventListener('click', async () => {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream, {
                        mimeType: 'audio/webm;codecs=pcm'
                    });
                    
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            audioChunks.push(event.data);
                            // Convert the audio data to base64 and send to the API
                            const reader = new FileReader();
                            reader.onloadend = () => {
                                const base64data = reader.result.split(',')[1];
                                sendAudioChunk(base64data);
                            };
                            reader.readAsDataURL(event.data);
                        }
                    };

                    mediaRecorder.onstop = () => {
                        stream.getTracks().forEach(track => track.stop());
                        audioChunks = [];
                    };

                    // Start recording with timeslices to get data periodically
                    mediaRecorder.start(1000);
                    isRecording = true;
                    startButton.disabled = true;
                    stopButton.disabled = false;
                    statusElement.textContent = 'Recording...';
                    
                    // Tell the server to start transcription
                    fetch('FASTAPI_URL_PLACEHOLDER/start_transcription', { method: 'POST' });
                    
                    // Update Streamlit
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: true
                    }, '*');
                    
                } catch (err) {
                    console.error('Error accessing microphone:', err);
                    statusElement.textContent = 'Error: ' + err.message;
                }
            });

            stopButton.addEventListener('click', () => {
                if (mediaRecorder && isRecording) {
                    mediaRecorder.stop();
                    isRecording = false;
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    statusElement.textContent = 'Recording stopped';
                    
                    // Tell the server to stop transcription
                    fetch('FASTAPI_URL_PLACEHOLDER/stop_transcription', { method: 'POST' });
                    
                    // Update Streamlit
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: false
                    }, '*');
                }
            });

            function sendAudioChunk(base64Audio) {
                fetch(apiUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ audio_data: base64Audio })
                })
                .catch(error => {
                    console.error('Error sending audio chunk:', error);
                });
            }

            // Cleanup when the component is unmounted
            window.addEventListener('beforeunload', () => {
                if (mediaRecorder && isRecording) {
                    mediaRecorder.stop();
                    fetch('FASTAPI_URL_PLACEHOLDER/stop_transcription', { method: 'POST' });
                }
            });
        </script>
    </body>
    </html>
    """
    
    # Replace the placeholder with the actual FastAPI URL
    fastapi_url = st.session_state.get('fastapi_url', 'https://realtimerag-1.onrender.com')
    audio_recorder_html = audio_recorder_html.replace('FASTAPI_URL_PLACEHOLDER', fastapi_url)
    
    # Render the HTML/JS component
    components.html(audio_recorder_html, height=150)
    
    return None
