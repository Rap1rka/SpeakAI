"""
SpeakAI - simple prototype (Gradio)
Features:
- Record voice from microphone (Gradio)
- Transcribe (OpenAI Whisper if OPENAI_API_KEY set; otherwise speech_recognition fallback)
- Send text to OpenAI chat (gpt-3.5-turbo) to simulate conversation partner + extract vocabulary
- Play response using pyttsx3 (offline TTS)
"""

import os
import io
import time
import tempfile
import openai
import gradio as gr
import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment

# ---------- Configuration ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 165)

recognizer = sr.Recognizer()

# ---------- Helper functions ----------

def convert_to_wav(file_bytes: bytes) -> str:
    """
    Gradio returns audio as WAV or other formats. Save to a temp WAV file and return path.
    Works cross-platform (Windows/Linux/Mac).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
        audio.export(tmp_path, format="wav")
    return tmp_path

def transcribe_with_openai(wav_path: str):
    """
    Use OpenAI Whisper (whisper-1) to transcribe audio.
    """
    with open(wav_path, "rb") as f:
        try:
            resp = openai.Audio.transcribe("whisper-1", f)
            text = resp.get("text", "").strip()
            return text
        except Exception as e:
            return f"[Transcription error with OpenAI: {e}]"

def transcribe_with_speechrecognition(wav_path: str):
    """
    Fallback transcription using speech_recognition (uses Google Web Speech by default).
    """
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            # NOTE: This uses Google Web Speech API (requires Internet) without an API key.
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "[Could not understand audio]"
        except sr.RequestError as e:
            return f"[Speech recognition service error: {e}]"

def ai_converse_and_vocab(user_text: str, conversation_history=None):
    """
    Send conversation to OpenAI chat model and also request a short vocab list with definitions.
    Returns: ai_reply, vocab_list (as text)
    """
    if conversation_history is None:
        conversation_history = []

    system_prompt = (
        "You are SpeakAI — a friendly, helpful English speaking partner and mini-tutor. "
        "When user speaks, respond naturally (short conversations), correct major pronunciation/grammar mistakes briefly, "
        "and provide friendly suggestions to improve fluency. After your conversational reply, return a short list (max 6) "
        "of important or new vocabulary words from the user's utterance or from the reply, with one-line definitions and an example sentence each. "
        "Format your response as JSON with keys 'reply' and 'vocab'. 'vocab' should be an array of objects with 'word','definition','example'."
    )

    messages = [{"role":"system","content": system_prompt}]
    for h in conversation_history:
        messages.append(h)
    messages.append({"role":"user","content": user_text})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=450,
            temperature=0.7,
        )
        ai_text = response["choices"][0]["message"]["content"]
    except Exception as e:
        ai_text = f"[OpenAI API error: {e}]"


    import json
    reply = ai_text
    vocab_text = ""
    try:
        start = ai_text.find("{")
        if start != -1:
            json_part = ai_text[start:]
            parsed = json.loads(json_part)
            reply = parsed.get("reply", "")
            vocab = parsed.get("vocab", [])
            vocab_lines = []
            for v in vocab:
                word = v.get("word", "")
                definition = v.get("definition", "")
                example = v.get("example", "")
                vocab_lines.append(f"{word} — {definition}. Example: {example}")
            vocab_text = "\n".join(vocab_lines)
        else:
            vocab_text = ""
    except Exception:
        vocab_text = ""

    return reply.strip(), vocab_text

def speak_text_tts(text: str):
    """
    Play the TTS (pyttsx3). This function returns immediately; in Gradio we'll provide also the text.
    """

    tts_engine.say(text)
    tts_engine.runAndWait()

# ---------- Gradio interface functions ----------

def process_audio(audio_file):
    """
    Main flow:
    1) convert audio bytes to wav
    2) transcribe (OpenAI Whisper if key present, else fallback)
    3) send user text to AI for reply + vocab
    4) return (user_text, ai_reply, vocab_text)
    """
    if audio_file is None:
        return "[No audio]", "", ""


    try:
        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
        else:
            audio_bytes = audio_file.read()
    except Exception:
        try:
            audio_bytes = audio_file[1].read()
        except Exception:
            return "[Could not read audio file]", "", ""

    wav_path = convert_to_wav(audio_bytes)

    if OPENAI_API_KEY:
        user_text = transcribe_with_openai(wav_path)
    else:
        user_text = transcribe_with_speechrecognition(wav_path)

    if OPENAI_API_KEY:
        ai_reply, vocab_text = ai_converse_and_vocab(user_text)
    else:
        ai_reply = (
            "This is an offline demo mode. I couldn't call OpenAI — please set OPENAI_API_KEY in environment "
            "to enable AI responses and vocab extraction."
        )
        vocab_text = ""

    try:
        import threading
        t = threading.Thread(target=speak_text_tts, args=(ai_reply,), daemon=True)
        t.start()
    except Exception:
        pass

    return user_text, ai_reply, vocab_text

# ---------- Build Gradio UI ----------

with gr.Blocks(title="SpeakAI") as demo:
    gr.Markdown("## SpeakAI — AI-powered speaking practice & smart dictionary")
    with gr.Row():
        with gr.Column(scale=1):

            gr.Image(value="A_presentation_slide_for_a_language_learning_appli.png", label="App logo (from slide)", visible=True)
        with gr.Column(scale=2):
            gr.Markdown("### How it works")
            gr.Markdown(
                "- Press **Start Recording** and speak (up to ~30 seconds).  \n"
                "- Your speech is transcribed, sent to AI which acts as a conversation partner and tutor.  \n"
                "- AI returns a short reply and a small vocab list with definitions.  \n"
                "- The reply is also played back using local TTS."
            )

    audio_input = gr.Audio(label="🎙️ Record your voice", format="wav", type="filepath", interactive=True)
    record_button = gr.Button("Process recording")

    user_text_out = gr.Textbox(label="Your transcribed text", lines=2)
    ai_reply_out = gr.Textbox(label="AI reply (and corrections)", lines=5)
    vocab_out = gr.Textbox(label="Vocabulary / Definitions", lines=6)

    def wrapper_process(audio_path):
        return process_audio(open(audio_path, "rb"))

    record_button.click(fn=process_audio, inputs=[audio_input], outputs=[user_text_out, ai_reply_out, vocab_out])

    gr.Markdown("**Note:** For best results set environment variable `OPENAI_API_KEY`. Without it the app uses local/limited transcription and disabled AI features.")

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
