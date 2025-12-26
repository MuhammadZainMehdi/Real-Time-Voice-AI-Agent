import os
import time
import io
import queue
import threading

import streamlit as st
import numpy as np
import sounddevice as sd
import whisper
from groq import Groq
from elevenlabs.client import ElevenLabs

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 5   # seconds per chunk

# ---------------- Globals (shared across reruns) ----------------
# These are module-level objects used by threads. They persist across Streamlit reruns.
audio_queue = queue.Queue()
transcript_queue = queue.Queue()
stop_event = threading.Event()

# You can reduce Whisper model size for faster startup (e.g. "tiny" / "base")
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

# Groq and ElevenLabs clients (reads keys from env)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
eleven_client = ElevenLabs(api_key=os.environ.get("ELEVEN_LABS"))

# ---------------- Backend thread functions (adapted) ----------------
def record():
    """Background recorder: captures 5s chunks and pushes numpy arrays into audio_queue."""
    # note: this runs in a background thread
    while not stop_event.is_set():
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        try:
            audio_queue.put_nowait(audio.flatten())
        except queue.Full:
            # if queue full, drop chunk (unlikely unless transcriber is very slow)
            pass

def transcribe_audio():
    """Background transcriber: consumes audio_queue, transcribes via Whisper, puts text into transcript_queue."""
    while not stop_event.is_set() or not audio_queue.empty():
        try:
            audio = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # Whisper expects a 1D numpy array or a file - we use the array
        try:
            result = model.transcribe(audio, fp16=False)
            text = result.get("text", "").strip()
            if text:
                transcript_queue.put_nowait(text)
        except Exception as e:
            # don't crash the thread — push an error message to transcript queue if you want to show it
            transcript_queue.put_nowait(f"[transcription error: {e}]")

# ---------------- LLM and TTS helpers (reuse your logic) ----------------
history = [
    {"role": "system", "content": "You are helpful assistant. Resolve user's query. Give concised answers."}
]

def generate_response(query):
    """Call Groq chat completion (blocking)."""
    history.append({"role": "user", "content": query})
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=history,
        temperature=0.2,
    )
    response = completion.choices[0].message.content
    history.append({"role": "assistant", "content": response})
    return response

def text_to_speech_bytes(text):
    """
    Convert text to mp3 bytes using ElevenLabs client.
    Returns bytes appropriate for st.audio(..., format='audio/mp3').
    """
    audio = eleven_client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    # `audio` should already be bytes-like (what your original code passed to play()).
    audio_bytes = b"".join(chunk for chunk in audio)
    return audio_bytes

# ---------------- Control functions for Streamlit ----------------
def start_recording():
    """Start recorder and transcriber threads and set session state flags."""
    # clear previous state & queues
    with suppress_queue_exceptions():
        clear_queue(audio_queue)
        clear_queue(transcript_queue)

    stop_event.clear()

    # threads
    recorder = threading.Thread(target=record, daemon=True)
    transcriber = threading.Thread(target=transcribe_audio, daemon=True)
    recorder.start()
    transcriber.start()

    # store thread refs in session_state so we can join later
    st.session_state['recorder_thread'] = recorder
    st.session_state['transcriber_thread'] = transcriber
    st.session_state['is_recording'] = True
    st.session_state['partial_transcript'] = []

def stop_recording():
    """Signal the threads to stop. We set flag so main loop finalizes."""
    stop_event.set()
    st.session_state['is_recording'] = False

def finalize_recording_and_process():
    """Join threads, collect transcripts, call LLM, get TTS bytes and return (transcript, response, audio_bytes)."""
    # Join threads if present
    recorder = st.session_state.get('recorder_thread')
    transcriber = st.session_state.get('transcriber_thread')
    if recorder:
        recorder.join(timeout=5)
    if transcriber:
        transcriber.join(timeout=5)

    # Drain transcript_queue
    collected = []
    while True:
        try:
            t = transcript_queue.get_nowait()
            collected.append(t)
        except queue.Empty:
            break

    # Merge with partial_transcript kept during live updates
    collected = st.session_state.get('partial_transcript', []) + collected
    full_text = " ".join(collected).strip()

    if not full_text:
        return collected, None, None

    # Call the LLM (blocking)
    response = generate_response(full_text)

    # Call TTS to get mp3 bytes (blocking)
    try:
        audio_bytes = text_to_speech_bytes(response)
    except Exception as e:
        audio_bytes = None
        st.warning(f"TTS failed: {e}")

    # clear stop_event to allow future recordings
    stop_event.clear()

    return collected, response, audio_bytes


# ---------------- Utility helpers ----------------
from contextlib import contextmanager

@contextmanager
def suppress_queue_exceptions():
    try:
        yield
    finally:
        pass

def clear_queue(q):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass
# Initialize session state
if 'partial_transcript' not in st.session_state:
    st.session_state['partial_transcript'] = []

# ---------------- Sidebar ----------------
st.sidebar.title("📊 Session Info")
num_transcripts = len(st.session_state['partial_transcript'])
st.sidebar.markdown(f"Transcripts: {num_transcripts}")

if st.sidebar.button("🧹 Clear History"):
    st.session_state['partial_transcript'] = []
    while not transcript_queue.empty():
        transcript_queue.get_nowait()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Voice AI Agent", page_icon= "🎤", layout="centered")
st.title("🎙️ Voice AI Agent")

# Initialize session state defaults
if 'is_recording' not in st.session_state:
    st.session_state['is_recording'] = False
if 'partial_transcript' not in st.session_state:
    st.session_state['partial_transcript'] = []
if 'recorder_thread' not in st.session_state:
    st.session_state['recorder_thread'] = None
if 'transcriber_thread' not in st.session_state:
    st.session_state['transcriber_thread'] = None

col1, col2 = st.columns([1, 3])

with col1:
    if not st.session_state['is_recording']:
        if st.button("▶️ Start Recording"):
            start_recording()
            # After starting, we will continue below into the "recording loop"
    else:
        # show active indicator and allow stop inside the live loop below
        st.markdown("**Recording...** 🔴")

with col2:
    st.markdown("**Live transcript**")
    transcript_box = st.empty()
    transcript_box.markdown("\n\n".join(st.session_state['partial_transcript']))

# If recording started this run, enter a short-loop to update the UI while still allowing Stop button press.
# This pattern shows Stop button inside the loop. Pressing Stop triggers a rerun which will set is_recording False.
if st.session_state['is_recording']:
    # The loop will run until user presses "Stop Recording" which calls stop_recording() via on_click in the button below.
    stop_btn = st.button("⏹ Stop Recording", on_click=stop_recording)

    # While recording, poll transcript_queue and update the transcript_box
    try:
        while st.session_state['is_recording']:
            updated = False
            while True:
                try:
                    txt = transcript_queue.get_nowait()
                except queue.Empty:
                    break
                st.session_state['partial_transcript'].append(txt)
                updated = True

            if updated:
                transcript_box.markdown("\n\n".join(st.session_state['partial_transcript']))

            # small sleep to avoid busy loop and give responsive UI
            time.sleep(0.25)
            # re-run the script to let Streamlit register button clicks (this is how Streamlit apps remain interactive)
            # The loop will exit when is_recording becomes False (stop_recording sets that on click)
            # Note: We do not call experimental_rerun; the run will continue and Streamlit will re-run on user interaction.
            # We just keep the loop short and responsive.
    except Exception as e:
        st.error(f"Recording loop error: {e}")
        stop_event.set()
        st.session_state['is_recording'] = False

# Once recording is stopped (either by user or something else), finalize and show the assistant response.
if (not st.session_state['is_recording']) and (st.session_state.get('recorder_thread') or st.session_state.get('transcriber_thread')):
    # Finalize: join threads and process remaining transcripts / call LLM/TTS
    with st.spinner("Finalizing recording, Wait for a moment"):
        transcripts, response, audio_bytes = finalize_recording_and_process()

    if transcripts:
        with st.chat_message("user"):
            st.markdown(" ".join(transcripts))
        
    else:
        st.info("No transcript collected.")

    if response:
        with st.chat_message("assistant"):
            st.markdown(response)


    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.info("No TTS audio available (or TTS failed).")

    # Reset thread references so next recording starts cleanly
    st.session_state['recorder_thread'] = None
    st.session_state['transcriber_thread'] = None
    # keep partial_transcript for user reference, user can clear it if desired
    if st.button("Clear transcript"):
        st.session_state['partial_transcript'] = []
