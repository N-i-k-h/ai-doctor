# If you don't use pipenv, you can uncomment:
# from dotenv import load_dotenv
# load_dotenv()

# Step1: (Optional) Microphone recording — NOT required for Gradio mic
# ffmpeg required for pydub export to mp3
import logging
import os
from io import BytesIO

import speech_recognition as sr
from pydub import AudioSegment
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Records from default microphone and saves to MP3.
    Not used by Gradio app (Gradio provides audio file path).
    """
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")

            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            logging.info(f"Audio saved to {file_path}")
    except Exception as e:
        logging.error(f"An error occurred recording audio: {e}")

def transcribe_with_groq(audio_filepath: str,
                         stt_model: str = "whisper-large-v3",
                         GROQ_API_KEY: str | None = None) -> str:
    """Transcribe an audio file using Groq Whisper."""
    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY missing. Set it in your .env")

    if not audio_filepath or not os.path.exists(audio_filepath):
        raise FileNotFoundError(f"Audio not found: {audio_filepath}")

    client = Groq(api_key=api_key)
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )
    return transcription.text.strip()

if __name__ == "__main__":
    # Example manual test (requires an mp3 file present)
    # print(transcribe_with_groq("patient_voice_test_for_patient.mp3"))
    pass
