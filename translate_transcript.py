import json
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=api_key, transport="rest")

def improve_translation(input_path="transcript.json", output_path="transcript_english.json"):
    if not os.path.exists(input_path):
        print("Error: Missing transcript.json")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = """
    You are an expert Telugu to English translator. 
    I am providing a JSON list of video subtitle segments. 
    Translate the 'text' value of each segment from Telugu to accurate, natural English.
    Ensure the psychological and educational context remains intact.
    DO NOT alter the 'id', 'start', or 'end' values. 
    Return ONLY a valid JSON array matching the exact input structure.
    """

    model = genai.GenerativeModel("gemini-3.7-flash")
    print("Sending Telugu transcript to Gemini for contextual translation...")
    
    response = None
    try:
        response = model.generate_content([prompt, json.dumps(data["segments"])])
        clean_json = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        translated_segments = json.loads(clean_json)
        
        data["segments"] = translated_segments
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Success: High-accuracy English transcript saved to {output_path}")
    except Exception as e:
        raw_output = response.text if response is not None else "No response generated (request failed before receiving a response)"
        print(f"Error parsing Gemini output: {e}\nRaw Output:\n{raw_output}")

if __name__ == "__main__":
    improve_translation()