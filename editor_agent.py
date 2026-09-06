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

GEMINI_MODEL_CANDIDATES = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash"
]

def generate_edit_plan(transcript_path=None, output_path="edit_plan.json"):
    """Analyzes the transcript and generates a JSON blueprint for high-retention reel editing."""
    if transcript_path is None:
        transcript_path = "transcript_english.json" if os.path.exists("transcript_english.json") else "transcript.json"

    if not os.path.exists(transcript_path):
        print(f"Error: {transcript_path} not found. Run transcribe.py first.")
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    prompt = """
    You are an elite short-form video director crafting viral, high-retention Reels, TikToks, and Shorts.
    This reference reel utilizes a high-retention listicle format, heavy outlined typography, and custom graphical illustrations instead of traditional cinematic stock footage.
    Analyze the provided timestamped transcript and return a strict JSON editing blueprint.

    YOUR EDITING STRATEGY:
    1. HOOK DETECTION:
       - Identify if there is a compelling, high-energy sentence or curiosity gap within the video that should serve as an upfront teaser hook (3 to 6 seconds max).
       - If the video already starts with a strong, natural opening, set "hook_segment": null.
    2. VISUAL AND GRAPHICAL OVERLAYS (Iconographic B-Roll):
       - Instead of traditional live-action stock video or generic photos, specifically select contextual 2D vector illustrations, minimalist line art, and symbolic icons.
       - Examples: "2d vector illustration of a weighing scale", "medical silhouette graphic", "stress brain icon animation", "symbolic line art of rejection letter", "minimalist icon graphic of corporate hierarchy", "gold weighing scale illustration".
       - Each asset must be tightly synchronized to appear instantly as the speaker transitions to each key concept or numbered point.
       - Always output English keywords for "search_keyword" optimized for vector/illustration/graphic queries.
       - Keep each visual appearance between 3.0 and 4.2 seconds.
    3. DYNAMIC TEXT CARDS & PUNCH-INS:
       - Identify high-retention listicle points or thematic concepts to pop up centrally on screen (1 to 2 seconds).
       - CRITICAL RULE: NEVER select speaker names, greetings, or self-introductions (e.g., 'Dr. Hema', 'I am...', 'My name is...', 'Psychologist'). Punch words must ONLY be punchy, thematic high-impact concepts (e.g., 'POOR COMMUNICATION', 'FEAR OF ENGLISH', 'REJECTED', 'JOB INTERVIEW', 'CONFIDENCE BLOCK').

    STRICT JSON OUTPUT SCHEMA:
    {
      "hook_segment": {"start": float, "end": float, "reason": "Why this hooks the viewer"} or null,
      "visual_cues": [
        {
          "start": float,
          "end": float,
          "search_keyword": "high-relevance 2d vector illustration or icon search term",
          "asset_type": "illustration",
          "reason": "Brief contextual rationale"
        }
      ],
      "punch_ins": [
        {"start": float, "end": float, "reason": "High-impact thematic concept (never speaker names/intros)"}
      ]
    }
    Return ONLY valid JSON.
    """

    print("Analyzing transcript and generating edit plan...")
    last_err = None
    for model_name in GEMINI_MODEL_CANDIDATES:
        try:
            print(f"Attempting editorial planning with model '{model_name}'...")
            model = genai.GenerativeModel(
                model_name,
                generation_config={"response_mime_type": "application/json"}
            )
            response = model.generate_content([prompt, json.dumps(transcript_data)])
            clean_json = response.text.strip().removeprefix('```json').removesuffix('```').strip()
            edit_plan = json.loads(clean_json)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(edit_plan, f, indent=2)

            print(f"Success: Edit plan saved to {output_path} using '{model_name}'")
            return edit_plan
        except Exception as e:
            last_err = e
            print(f"Model '{model_name}' error: {e}. Trying next fallback candidate...")
            continue

    print(f"Failed to generate edit plan across all models: {last_err}")
    return None

if __name__ == "__main__":
    generate_edit_plan()