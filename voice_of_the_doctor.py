# ===============================
# AI Doctor Voice (gTTS + ElevenLabs)
# ===============================
import os
import subprocess
import platform
from gtts import gTTS
from dotenv import load_dotenv
from elevenlabs import save
from elevenlabs.client import ElevenLabs

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def _play_audio(file_path: str):
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(["afplay", file_path], check=False)
        elif os_name == "Windows":
            subprocess.run(["powershell", "-c",
                            f'(New-Object Media.SoundPlayer "{file_path}").PlaySync();'], check=False)
        elif os_name == "Linux":
            # mpg123 / ffplay also work if installed
            subprocess.run(["aplay", file_path], check=False)
    except Exception as e:
        print(f"[WARN] Could not auto-play audio: {e}")

# ---- Names EXACTLY as imported in gradio_app.py ----
def text_to_speech_with_gtts(input_text: str, output_filepath: str = "gtts_output.mp3", autoplay: bool = False) -> str:
    """Convert text to speech using gTTS. Returns saved file path."""
    try:
        tts = gTTS(text=input_text, lang="en", slow=False)
        tts.save(output_filepath)
        print(f"[gTTS] Audio saved to {output_filepath}")
        if autoplay:
            _play_audio(output_filepath)
        return output_filepath
    except Exception as e:
        raise RuntimeError(f"gTTS failed: {e}")

def text_to_speech_with_elevenlabs(input_text: str, output_filepath: str = "elevenlabs_output.mp3",
                                   voice: str = "Aria", autoplay: bool = False) -> str:
    """Convert text to speech using ElevenLabs. Returns saved file path."""
    if not ELEVENLABS_API_KEY:
        raise EnvironmentError("ELEVENLABS_API_KEY not found in environment variables!")

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio = client.generate(
            text=input_text,
            voice=voice,               # use your preferred voice, e.g. "Rachel"
            model="eleven_turbo_v2",
            output_format="mp3_22050_32",
        )
        save(audio, output_filepath)
        print(f"[ElevenLabs] Audio saved to {output_filepath}")
        if autoplay:
            _play_audio(output_filepath)
        return output_filepath
    except Exception as e:
        raise RuntimeError(f"ElevenLabs TTS failed: {e}")

if __name__ == "__main__":
    sample = "Hello! This is the AI Doctor speaking."
    print(text_to_speech_with_gtts(sample, "doctor_gtts.mp3"))
    print(text_to_speech_with_elevenlabs(sample, "doctor_elevenlabs.mp3", voice="Rachel"))
