import json
import os
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

# Use environment variable or .env
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    print("Warning: GEMINI_API_KEY not set in environment or .env file.")

genai.configure(api_key=api_key, transport="rest")

def generate_edit_plan(transcript_path=None, output_path="edit_plan.json"):
    """Analyzes the transcript and generates a JSON blueprint for the video editor."""
    if transcript_path is None:
        transcript_path = "transcript_english.json" if os.path.exists("transcript_english.json") else "transcript.json"

    if not os.path.exists(transcript_path):
        print(f"Error: {transcript_path} not found. Run transcribe.py first.")
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    prompt = """
    You are an award-winning social video editor creating viral, engaging short-form videos (Reels, TikTok, Shorts, YouTube).
    Analyze the provided timestamped transcript and return a strict JSON editing blueprint.
    YOUR EDITING STRATEGY:
    1. HOOK DETECTION:
    - Identify if there is a compelling, high-energy sentence or curiosity gap within the video that should serve as an upfront teaser hook (3 to 6 seconds max).
    - If the video already starts with a strong, natural opening, set "hook_segment": null.
    2. VISUAL ASSETS (Pictures & B-Roll Cues):
    - Identify key moments where a visual aid enhances comprehension or retention.
    - DO NOT restrict yourself to literal physical nouns. Include conceptual, metaphorical, or emotional visuals (e.g., "market crash graph", "frustrated developer", "serene morning nature", "cryptocurrency wallet", "celebration confetti").
    - Always output English keywords for "search_keyword" (optimized for stock photo/video APIs).
    - Keep each visual appearance between 4.0 and 4.5 seconds. Space them out naturally based on content density.
    3. PUNCH-INS (Dynamic Zooms):
    - Identify 2 to 4 high-emphasis words, surprising statistics, or punchlines to zoom in (1.15x) for 1 to 2 seconds to reset viewer attention.
    STRICT JSON OUTPUT SCHEMA:
    {
    "hook_segment": {"start": float, "end": float, "reason": "Why this hooks the viewer"} or null,
    "visual_cues": [
        {
        "start": float,
        "end": float,
        "search_keyword": "high-relevance English stock photo search term",
        "asset_type": "photo",
        "reason": "Brief contextual rationale"
        }
    ],
    "punch_ins": [
        {"start": float, "end": float, "reason": "Emphasis punchline"}
    ]
    }
    Return ONLY valid JSON.
    """

    # Using gemini-3.6-flash with structured JSON output
    model = genai.GenerativeModel(
        "gemini-3.6-flash",
        generation_config={"response_mime_type": "application/json"}
    )
    
    print("Analyzing transcript and generating edit plan...")
    response = model.generate_content([prompt, json.dumps(transcript_data)])
    
    try:
        # Clean potential markdown formatting from the response
        clean_json = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        edit_plan = json.loads(clean_json)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(edit_plan, f, indent=2)
            
        print(f"Success: Edit plan saved to {output_path}")
    except Exception as e:
        print(f"Failed to parse LLM output: {e}\nRaw Output:\n{response.text}")

if __name__ == "__main__":
    generate_edit_plan()