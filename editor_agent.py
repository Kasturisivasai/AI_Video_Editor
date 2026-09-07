import json
import os
import google.generativeai as genai

from dotenv import load_dotenv
from pipeline import robust_json_parse, normalize_timestamp

load_dotenv()

# Use environment variable or .env
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    print("Warning: GEMINI_API_KEY not set in environment or .env file.")

genai.configure(api_key=api_key, transport="rest")

GEMINI_MODEL_CANDIDATES = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash"
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
    You are an elite, award-winning short-form video director creating viral, broadcast-quality Reels, TikToks, and YouTube Shorts (in the style of high-end documentary creators like Ali Abdaal, Vox, and Alex Hormozi).
    Analyze the provided timestamped transcript and create a strict JSON editing blueprint that is hyper-relatable, engaging, and cinematic.

    YOUR EDITING STRATEGY:
    1. HOOK DETECTION:
       - Identify if there is a compelling, high-energy sentence or curiosity gap within the video that should serve as an upfront teaser hook (3 to 6 seconds max).
       - If the video already starts with a strong, natural opening, set "hook_segment": null.

    2. HYPER-RELATABLE, CINEMATIC B-ROLL OVERLAYS:
       - GENERATE EXACTLY 7 TO 9 VISUAL CUES distributed evenly across the video timeline (approx every 9 to 12 seconds).
       - NARRATIVE MATCHING (STORY-DRIVEN RELEVANCE):
         * Every visual cue MUST depict the EXACT real-world action, emotion, or situation the speaker is narrating at that second.
         * If the speaker tells a story about a student studying hard & getting marks: show a dedicated student focused on books and laptop in a library.
         * If the speaker talks about fear of speaking English or interview anxiety: show a nervous candidate waiting in an office or sitting at an interview desk.
         * If the speaker talks about repeated interview rejections or struggles: show a thoughtful person looking out of an office window or reflecting over notes.
         * If the speaker talks about practicing communication: show two professionals in a natural, candid discussion across a desk or over coffee.
         * If the speaker talks about confidence or success: show a confident handshake or a clear professional presentation.

       - STRICT AESTHETIC RULES — ZERO CHEESY STOCK PHOTOS:
         * ABSOLUTELY FORBIDDEN to use generic stock models grinning or smiling directly into the camera lens! (NO 'smiling at camera', NO 'thumbs up', NO fake posed corporate grins).
         * ALL scenes MUST BE CANDID, ACTION-ORIENTED, and CINEMATIC:
           - Subjects must be actively doing something (typing, writing notes, conversing naturally, listening attentively, walking with purpose, looking thoughtful).
           - Modern aesthetic: candid documentary angle, natural workplace or study lighting, shallow depth of field.
         * NO abstract vectors, clipart, 3D icons, or hand signs (NEVER use 'symbol', 'icon', 'vector', 'illustration', 'gesture').

       - DYNAMIC ASSET TYPE:
         * Mix 'video' and 'photo'.
         * Use 'video' for human actions, conversations, studying, and movement (vertical video clips feel 10x more dynamic).
         * Use 'photo' for thoughtful still moments.

       - SEARCH KEYWORD FORMAT:
         * Keep search_keyword to 3 to 5 concise, punchy photographic words (e.g. 'candid job interview desk', 'focused student library laptop', 'thoughtful professional office window', 'two colleagues talking desk', 'confident handshake office').
       - Keep each visual appearance between 3.0 and 4.5 seconds.

    3. DYNAMIC CENTER PUNCH CALLOUTS (High-Impact Power Words):
       - Identify 5 to 7 high-impact power concepts across the timeline (1.5 to 2.2 seconds each).
       - STRICT RULES FOR CALLOUT WORDS:
         * EXACTLY 1 TO 2 WORDS MAXIMUM.
         * MUST be core thematic concepts or strong emotional triggers (e.g., 'POOR COMMUNICATION', 'GREAT MARKS', 'ENGLISH FEAR', 'FAILED REPEATEDLY', 'HAVE KNOWLEDGE', 'HOW TO PRACTICE').
         * NEVER select grammatical filler words (FORBIDDEN: 'THEY MUST', 'FOR GIVING', 'LEARN YOUR', 'WE ARE', 'CAN BE', 'IN THE', 'SO THAT', 'IT IS', 'BECAUSE OF', 'ABOUT THIS').
         * NEVER select speaker names or titles ('HEMA', 'HYMA PRASAD', 'DOCTOR', 'PSYCHOLOGIST', 'MYSELF').

    STRICT JSON OUTPUT SCHEMA:
    {
      "hook_segment": {"start": float, "end": float, "reason": "Why this hooks the viewer"} or null,
      "visual_cues": [
        {
          "start": float,
          "end": float,
          "search_keyword": "3-5 word candid, action-focused search query (e.g. 'candid job interview office desk')",
          "asset_type": "video" or "photo",
          "display_mode": "fullscreen",
          "reason": "Direct narrative link to the spoken words"
        }
      ],
      "punch_ins": [
        {
          "start": float,
          "end": float,
          "callout_text": "EXACT 1-2 POWER WORDS (e.g. 'ENGLISH FEAR', 'GREAT MARKS')",
          "reason": "Why this concept hits hard"
        }
      ]
    }
    CRITICAL: Output strict standard JSON without comments, single quotes, or trailing commas.
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
            edit_plan = robust_json_parse(response.text)

            # Sanitize and clamp all cues and punch ins
            segs = transcript_data.get("segments", []) if isinstance(transcript_data, dict) else []
            total_dur = transcript_data.get("duration") or (segs[-1]["end"] if segs else 105.0)
            
            clean_cues = []
            for cue in edit_plan.get("visual_cues", []):
                c_start = normalize_timestamp(cue.get("start", 0), max_duration=total_dur)
                c_end = normalize_timestamp(cue.get("end", 0), max_duration=total_dur)
                if c_start < total_dur - 0.5:
                    c_end = min(c_end, total_dur)
                    if c_end <= c_start:
                        c_end = round(min(c_start + 3.5, total_dur), 2)
                    cue["start"] = c_start
                    cue["end"] = c_end
                    clean_cues.append(cue)
            edit_plan["visual_cues"] = clean_cues

            clean_punches = []
            for pi in edit_plan.get("punch_ins", []):
                p_start = normalize_timestamp(pi.get("start", 0), max_duration=total_dur)
                p_end = normalize_timestamp(pi.get("end", 0), max_duration=total_dur)
                if p_start < total_dur - 0.5:
                    p_end = min(p_end, total_dur)
                    if p_end <= p_start:
                        p_end = round(min(p_start + 2.0, total_dur), 2)
                    pi["start"] = p_start
                    pi["end"] = p_end
                    clean_punches.append(pi)
            edit_plan["punch_ins"] = clean_punches

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