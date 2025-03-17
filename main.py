from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
import queue
import uvicorn
import base64
import io
import numpy as np
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from asyncio import Semaphore
from ragEmbed import async_update_db
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

class AudioData(BaseModel):
    audio_data: str  # Base64 encoded audio data

app = FastAPI()

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

audio_queue = queue.Queue()
transcription_active = False
current_transcription = ""
transcription_task = None

class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream):
        super().__init__(output_stream)
        self.current_words = []
        self.chunk_size = 200
        self.overlap_size = 70
        self.previous_chunk_end = []
        self.last_transcript = ""
        self.upsert_sem = Semaphore(5)
        
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        global current_transcription
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                new_text = alt.transcript
                
                if new_text.startswith(self.last_transcript):
                    new_text = new_text[len(self.last_transcript):].strip()
                new_words = new_text.split()
                if new_words:
                    self.current_words.extend(new_words)
                    self.last_transcript = alt.transcript
                    # Update the current transcription for the get endpoint
                    current_transcription = new_text
                if len(self.current_words) >= self.chunk_size:
                    await self.store_chunk()
                    
    async def store_chunk(self):
        if len(self.current_words) < self.chunk_size:
            return
        chunk_start = max(0, len(self.previous_chunk_end) - self.overlap_size)
        overlap = self.previous_chunk_end[chunk_start:]
        new_chunk_words = overlap + self.current_words[:self.chunk_size]
        self.previous_chunk_end = new_chunk_words[-self.overlap_size:]
        self.current_words = self.current_words[self.chunk_size:]
        chunk_text = " ".join(new_chunk_words)
        asyncio.create_task(self.upsert_to_vector_db(chunk_text))
        
    async def final_flush(self):
        if self.current_words:
            chunk_start = max(0, len(self.previous_chunk_end) - self.overlap_size)
            chunk_text = " ".join(self.previous_chunk_end[chunk_start:] + self.current_words)
            await self.upsert_to_vector_db(chunk_text)
            
    async def upsert_to_vector_db(self, chunk):
        async with self.upsert_sem:
            try:
                await async_update_db(chunk)
            except Exception as e:
                print(f"Failed to upsert: {str(e)}")

async def write_chunks(stream):
    global transcription_active
    while transcription_active:
        if not audio_queue.empty():
            chunk = audio_queue.get()
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        else:
            await asyncio.sleep(0.1)  # Small sleep to prevent CPU hogging
    await stream.input_stream.end_stream()

async def basic_transcribe():
    global transcription_active
    client = TranscribeStreamingClient(region="us-east-1")
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm"
    )
    handler = MyEventHandler(stream.output_stream)
    try:
        await asyncio.gather(
            write_chunks(stream),
            handler.handle_events(),
        )
    finally:
        await handler.final_flush()
        await stream.input_stream.end_stream()

@app.get("/")
async def root():
    return {"message": "FastAPI server is running!"}

@app.post("/start_transcription")
async def start_transcription():
    global transcription_active, transcription_task, current_transcription
    
    if transcription_active:
        return {"message": "Transcription already active"}
    
    transcription_active = True
    current_transcription = ""
    transcription_task = asyncio.create_task(basic_transcribe())
    return {"message": "Transcription started"}

@app.post("/stop_transcription")
async def stop_transcription():
    global transcription_active, transcription_task
    
    if not transcription_active:
        return {"message": "No active transcription"}
    
    transcription_active = False
    if transcription_task:
        try:
            await asyncio.wait_for(transcription_task, timeout=5.0)
        except asyncio.TimeoutError:
            pass  # Task didn't complete in time, but we're stopping anyway
        transcription_task = None
    
    return {"message": "Transcription stopped"}

@app.get("/get_transcription")
async def get_transcription():
    global current_transcription
    response = {"transcription": current_transcription}
    current_transcription = ""  # Reset after reading
    return JSONResponse(content=response)

@app.post("/send_audio")
async def send_audio(audio_data: AudioData):
    if not transcription_active:
        return {"message": "Transcription not active"}
    
    try:
        # Decode base64 audio data
        decoded_data = base64.b64decode(audio_data.audio_data)
        
        # Convert to 16-bit PCM format (what Amazon Transcribe expects)
        # This assumes the incoming audio is already in the right format
        # You may need to adjust this based on your audio capture format
        audio_bytes = io.BytesIO(decoded_data).read()
        
        # Put the audio data in the queue
        audio_queue.put(audio_bytes)
        
        return {"message": "Audio data received"}
    except Exception as e:
        return {"message": f"Error processing audio: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
