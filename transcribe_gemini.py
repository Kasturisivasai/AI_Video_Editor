import json
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=api_key, transport="rest")

def transcribe_with_gemini(audio_path="temp_audio.wav", output_path="transcript_english.json"):
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return

    print("Reading audio file...")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    model = genai.GenerativeModel("gemini-3.6-flash", generation_config={"response_mime_type": "application/json"})
    
    prompt = """
    Listen to this entire audio track carefully.
    The speaker is Dr. Hymavathi, speaking in Telugu about communication skills, interview fear, and psychological confidence.
    
    Translate and transcribe the entire speech into concise, natural English subtitles.
    Guidelines for high-engagement subtitles:
    - Break sentences into short, readable phrases (3 to 6 words each).
    - Each segment should have accurate start and end timestamps in seconds (e.g. 1.2s to 3.0s).
    - Cover the entire speech timeline continuously from 0s to 90s without gaps or skipping.
    
    Return strict JSON matching this schema:
    {
      "language": "te",
      "segments": [
        {"id": 1, "start": float, "end": float, "text": "English subtitle phrase"}
      ]
    }
    """

    print("Transcribing and translating audio with Gemini 3.6 Flash...")
    response = model.generate_content([{"mime_type": "audio/wav", "data": audio_bytes}, prompt])
    
    try:
        clean_json = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(clean_json)
        if isinstance(parsed, list):
            data = {"language": "te", "segments": parsed}
        else:
            data = parsed
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        print(f"Success: Saved {len(data.get('segments', []))} subtitle segments to {output_path}")
        return data
    except Exception as e:
        print(f"Error parsing Gemini response: {e}\nRaw text: {response.text[:500]}")
        return None

if __name__ == "__main__":
    transcribe_with_gemini()
