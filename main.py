from fastapi import FastAPI, BackgroundTasks
import asyncio
import queue
import numpy as np
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from asyncio import Semaphore
from ragEmbed import async_update_db

app = FastAPI()
audio_queue = queue.Queue()

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
                
                if new_text.startswith(self.last_transcript):
                    new_text = new_text[len(self.last_transcript):].strip()

                new_words = new_text.split()
                if new_words:
                    self.current_words.extend(new_words)
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
            except Exception as e:
                print(f"Failed to upsert: {str(e)}")

async def write_chunks(stream):
    while True:
        if not audio_queue.empty():
            chunk = audio_queue.get()
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
    await stream.input_stream.end_stream()

async def basic_transcribe():
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

@app.post("/transcribe/start")
def start_transcription(background_tasks: BackgroundTasks):
    background_tasks.add_task(asyncio.run, basic_transcribe())
    return {"message": "Transcription started"}
