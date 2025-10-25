# If you don't use pipenv uncomment these two lines:
# from dotenv import load_dotenv
# load_dotenv()

import os
import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

SYSTEM_PROMPT = (
    "You have to act as a professional doctor, this is for learning purpose. "
    "With what I see, tell me if there is anything medically concerning. "
    "If you propose a differential, suggest brief remedies. "
    "Do not add numbers or special characters. Reply in one short paragraph only "
    "as if you are speaking to a real person. Keep it concise (max two sentences). "
    "Start directly without preamble."
)

DEFAULT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # will auto-fallback if unavailable


def process_inputs(audio_filepath, image_filepath):
    # 1) Speech -> Text
    try:
        speech_text = transcribe_with_groq(
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3",
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
        )
    except Exception as e:
        speech_text = f"[STT error] {e}"

    # 2) Image + Query -> Doctor text
    if image_filepath:
        try:
            enc = encode_image(image_filepath)
            doctor_response = analyze_image_with_query(
                query=f"{SYSTEM_PROMPT} {speech_text}",
                encoded_image=enc,
                model=DEFAULT_VISION_MODEL,
            )
        except Exception as e:
            doctor_response = f"[Vision error] {e}"
    else:
        doctor_response = "No image provided for me to analyze"

    # 3) Doctor text -> Voice (try ElevenLabs, fallback to gTTS)
    audio_out_path = "final.mp3"
    voice_path = None  # default

    try:
        voice_path = text_to_speech_with_elevenlabs(
            input_text=doctor_response,
            output_filepath=audio_out_path,
        )
    except Exception as e:
        print(f"[WARN] ElevenLabs failed: {e}. Falling back to gTTS...")
        try:
            voice_path = text_to_speech_with_gtts(
                input_text=doctor_response,
                output_filepath=audio_out_path,
            )
        except Exception as e2:
            print(f"[ERROR] gTTS also failed: {e2}")
            voice_path = None

    # ✅ Ensure audio file is valid before returning
    if not voice_path or not os.path.exists(voice_path) or os.path.getsize(voice_path) == 0:
        voice_path = None

    return speech_text, doctor_response, voice_path


iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Speak (microphone)"),
        gr.Image(type="filepath", label="Upload/Drop an image"),
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice"),
    ],
    title="AI Doctor with Vision and Voice",
    description="Speak your concern, upload a photo, get a concise doctor-style reply and a voice response.",
)


if __name__ == "__main__":
    # --- THIS IS THE UPDATED PART ---
    # Get the port from the environment variable (Render sets this)
    port = int(os.environ.get("PORT", 7860))
    # Launch the app to be accessible externally (0.0.0.0)
    iface.launch(server_name="0.0.0.0", server_port=port)
