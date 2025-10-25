import os
import base64
from dotenv import load_dotenv
from groq import Groq
from elevenlabs import save
from elevenlabs.client import ElevenLabs

# Load environment variables
load_dotenv()

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize clients (lazy init in functions too, but keep here for quick fails)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
tts_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None

def encode_image(image_path: str) -> str:
    """Return base64 string for an image file."""
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image_with_query(query: str, model: str, encoded_image: str) -> str:
    """Send a multimodal query (text + image) to Groq."""
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is missing. Set it in your .env")

    client = groq_client or Groq(api_key=GROQ_API_KEY)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ],
        }
    ]

    # If your chosen model is unavailable, try a vision-capable fallback
    tried = []
    for m in [model, "llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]:
        if not m:
            continue
        try:
            tried.append(m)
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=m,
                temperature=0.2,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Groq vision call failed. Tried models: {tried}. Last error: {last_err}")

def speak_text(text: str, voice: str = "Rachel", output_file: str = "doctor_voice.mp3") -> str:
    """Synthesize speech with ElevenLabs; returns saved path."""
    if not ELEVENLABS_API_KEY:
        raise EnvironmentError("ELEVENLABS_API_KEY is missing. Set it in your .env")

    client = tts_client or ElevenLabs(api_key=ELEVENLABS_API_KEY)
    # Model/voice names may vary per account/region
    audio = client.generate(
        text=text,
        voice=voice,
        model="eleven_turbo_v2",
        output_format="mp3_22050_32",
    )
    save(audio, output_file)
    print(f"✅ Doctor’s voice saved to {output_file}")
    return output_file

# Example usage
if __name__ == "__main__":
    query = "Is there something wrong with my face?"
    model = "meta-llama/llama-4-scout-17b-16e-instruct"  # will auto-fallback if unavailable
    image_path = "acne.jpg"

    enc = encode_image(image_path)
    result = analyze_image_with_query(query, model, enc)
    print("Doctor says:", result)
    speak_text(result, voice="Rachel")
