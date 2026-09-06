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
    You are an award-winning short-form video director creating viral, broadcast-quality Reels, TikToks, and Shorts.
    Analyze the provided timestamped transcript and return a strict JSON editing blueprint.

    YOUR EDITING STRATEGY:
    1. HOOK DETECTION:
       - Identify if there is a compelling, high-energy sentence or curiosity gap within the video that should serve as an upfront teaser hook (3 to 6 seconds max).
       - If the video already starts with a strong, natural opening, set "hook_segment": null.

    2. VISUAL AND B-ROLL OVERLAYS (High-Quality Stock Imagery / Video):
       - GENERATE EXACTLY 8 TO 9 VISUAL CUES distributed evenly across the video timeline (approx every 8 to 12 seconds).
       - Each asset must tightly synchronize to appear as the speaker introduces a key symptom, concept, workplace situation, or numbered point.
       - SEARCH KEYWORDS: Write clean, evocative, high-resolution stock photography and video search terms (e.g. "stressed businessman headache office", "insomnia alarm clock dark bedroom", "exhausted doctor burnout moody lighting", "stomach pain gastritis healthy diet", "professional counseling therapy session", "employee burnout laptop desk").
       - DO NOT write "2d vector illustration" or "icon graphic" in search keywords, as stock libraries return low-resolution clipart. Use realistic, cinematic stock photography/video search terms.
       - Keep each visual appearance between 3.0 and 4.0 seconds.

    3. DYNAMIC CENTER PUNCH CALLOUTS (High-Impact Power Words):
       - Identify 5 to 7 high-impact power words or punch concepts across the timeline (1.5 to 2.2 seconds each).
       - STRICT RULES FOR CALLOUT WORDS:
         * EXACTLY 1 TO 2 WORDS MAXIMUM.
         * MUST be core thematic concepts, strong emotions, or critical terms (e.g., 'BURNOUT', 'INSOMNIA', 'JOB STRESS', 'GASTRITIS', 'DEPRESSION', 'REJECTION', 'POOR SALARY', 'ANXIETY', 'COMMUNICATION').
         * NEVER select grammatical filler words, auxiliary verbs, prepositions, or pronouns (STRICTLY FORBIDDEN: 'THEY MUST', 'FOR GIVING', 'LEARN YOUR', 'WE ARE', 'CAN BE', 'IN THE', 'SO THAT', 'IT IS', 'BECAUSE OF', 'ABOUT THIS').
         * NEVER select speaker names, greetings, or self-introductions (STRICTLY FORBIDDEN: 'HEMA', 'HYMA PRASAD', 'DOCTOR', 'PSYCHOLOGIST', 'MYSELF').

    STRICT JSON OUTPUT SCHEMA:
    {
      "hook_segment": {"start": float, "end": float, "reason": "Why this hooks the viewer"} or null,
      "visual_cues": [
        {
          "start": float,
          "end": float,
          "search_keyword": "clean high-resolution stock photo/video search query (e.g. stressed executive headache desk)",
          "asset_type": "photo",
          "display_mode": "fullscreen",
          "reason": "Contextual rationale"
        }
      ],
      "punch_ins": [
        {
          "start": float,
          "end": float,
          "callout_text": "EXACT 1-2 POWER WORDS (e.g. 'BURNOUT', 'INSOMNIA')",
          "reason": "Why this concept hits hard"
        }
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