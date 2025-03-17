import asyncio
import queue
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from asyncio import Semaphore
from ragEmbed import async_update_db

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Queue for audio frames
audio_queue = queue.Queue()
semaphore = Semaphore(1)  # Prevent multiple transcriptions at once

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
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                new_text = alt.transcript
                
                # Avoid repeating words
                if new_text.startswith(self.last_transcript):
                    new_text = new_text[len(self.last_transcript):].strip()

                new_words = new_text.split()
                if new_words:
                    self.current_words.extend(new_words)
                    st.write("[Streaming]:", new_text)  # Display in Streamlit UI
                    self.last_transcript = alt.transcript

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
                st.write(f"[Upserted] {chunk[:50]}...")
            except Exception as e:
                st.error(f"Failed to upsert: {str(e)}")

def audio_callback(frame):
    """Handles audio frames from WebRTC and stores in queue"""
    audio_data = np.frombuffer(frame.to_ndarray(), dtype=np.int16)
    audio_queue.put(audio_data.tobytes())
    return frame  # Return the frame unmodified

async def write_chunks(stream):
    """Continuously sends audio chunks to AWS Transcribe"""
    while st.session_state["transcription_running"]:
        if not audio_queue.empty():
            chunk = audio_queue.get()
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await asyncio.sleep(0.1)  # Avoid tight loop

    await stream.input_stream.end_stream()  # Stop stream when stopped

async def basic_transcribe():
    """Handles real-time transcription with start/stop functionality."""
    if "transcription_running" not in st.session_state:
        st.session_state["transcription_running"] = False  # Default state

    async with semaphore:  # Prevent multiple transcriptions
        client = TranscribeStreamingClient(region="us-east-1")
        stream = await client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=16000,
            media_encoding="pcm"
        )
        handler = MyEventHandler(stream.output_stream)

        try:
            transcription_task = asyncio.create_task(write_chunks(stream))
            event_task = asyncio.create_task(handler.handle_events())

            while st.session_state["transcription_running"]:
                await asyncio.sleep(0.1)  # Check stop flag periodically

            transcription_task.cancel()
            event_task.cancel()

        except asyncio.CancelledError:
            pass  # Prevents errors when canceling tasks

        finally:
            await handler.final_flush()  # Save transcript to DB
            await stream.input_stream.end_stream()  # Close stream

# WebRTC Streamer
webrtc_ctx = webrtc_streamer(
    key="audio-only",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    audio_frame_callback=audio_callback,
    video_frame_callback=None,
    media_stream_constraints={"video": False, "audio": True},
)

# UI Buttons to start/stop transcription
if st.button("Start Transcription"):
    st.session_state["transcription_running"] = True
    asyncio.create_task(basic_transcribe())

if st.button("Stop Transcription"):
    st.session_state["transcription_running"] = False
