import glob
import json
import os
import re
import shutil
import subprocess
import requests
import PIL.Image
import numpy as np

# Patch MoviePy 1.0.3 with Pillow >= 10.0
if not hasattr(PIL.Image, "ANTIALIAS"):
    setattr(PIL.Image, "ANTIALIAS", PIL.Image.Resampling.LANCZOS)

import google.generativeai as genai
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, ColorClip
from moviepy.video.fx.crop import crop
import imageio_ffmpeg
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_PEXELS_KEY = os.environ.get("PEXELS_API_KEY", "")

def get_ffmpeg_exe():
    """Finds the FFmpeg executable."""
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()

def extract_audio(video_path: str, audio_path: str = "temp_audio.wav") -> str:
    """Extracts a 16kHz mono WAV audio file from the input video."""
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return audio_path

GEMINI_MODEL_CANDIDATES = [
    "gemini-3.7-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.6-flash"
]

def transcribe_audio_gemini(audio_path: str, api_key: str = DEFAULT_GEMINI_KEY, vocab_hints: str = "") -> dict:
    """Uses resilient Gemini models with auto-fallback to transcribe and translate audio into concise English subtitle segments."""
    genai.configure(api_key=api_key, transport="rest")
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    hint_text = ""
    if vocab_hints and vocab_hints.strip():
        hint_text = f"""
    IMPORTANT PROPER NOUN & VOCABULARY HINTS:
    The following names, proper nouns, or specialized terminology are mentioned in this video:
    "{vocab_hints.strip()}"
    Always prioritize and use these exact spellings in your English subtitle transcription.
    """

    prompt = f"""
    Listen to this entire audio track carefully.
    Transcribe and translate the entire speech into natural, conversational English subtitle segments.
    {hint_text}
    CRITICAL GUIDELINES FOR HIGH-ENGAGEMENT SOCIAL CAPTIONS:
    - Keep each segment short and punchy: strictly 3 to 6 words (under 30 characters).
    - Accurately timestamp start and end in seconds (e.g., 0.8 to 4.2).
    - Cover the entire speech timeline from beginning to end without gaps or skipping.
    
    Return strict JSON matching this schema:
    {{
      "language": "detected_code",
      "segments": [
        {{"id": 1, "start": float, "end": float, "text": "Short punchy phrase"}}
      ]
    }}
    """
    last_err = None
    for model_name in GEMINI_MODEL_CANDIDATES:
        try:
            print(f"Attempting transcription with model '{model_name}'...")
            model = genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json"})
            response = model.generate_content([{"mime_type": "audio/wav", "data": audio_bytes}, prompt])
            clean_json = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(clean_json)
            if isinstance(data, list):
                data = {"segments": data}
            print(f"Transcription successful using model '{model_name}' ({len(data.get('segments', []))} segments)")
            return data
        except Exception as e:
            last_err = e
            print(f"Model '{model_name}' quota/error: {e}. Falling back to next candidate...")
            continue
    raise last_err

def generate_edit_plan_gemini(transcript_data: dict, api_key: str = DEFAULT_GEMINI_KEY) -> dict:
    """AI Director Agent: Analyzes transcript to select 8-9 high-relevance visual cues and 1-2 word power punch callouts with auto-fallback."""
    genai.configure(api_key=api_key, transport="rest")

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
    last_err = None
    for model_name in GEMINI_MODEL_CANDIDATES:
        try:
            print(f"Attempting editorial planning with model '{model_name}'...")
            model = genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json"})
            response = model.generate_content([prompt, json.dumps(transcript_data)])
            clean_json = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(clean_json)
            print(f"Editorial plan generated successfully using model '{model_name}' ({len(data.get('visual_cues', []))} visual cues, {len(data.get('punch_ins', []))} punch callouts)")
            return data
        except Exception as e:
            last_err = e
            print(f"Model '{model_name}' planning error: {e}. Falling back to next candidate...")
            continue
    raise last_err

ASS_COLOR_MAP = {
    "Pure White": "&H00FFFFFF",
    "Electric Gold": "&H0000E6FF",
    "Neon Yellow": "&H0000FFFF",
    "Crimson Red": "&H002020FF",
    "Neon Cyan": "&H00FFFF00",
    "Soft Yellow": "&H0075F2FF",
    "Lime Green": "&H0032FF32",
    "white": "&H00FFFFFF",
    "gold": "&H0000E6FF",
    "red": "&H002020FF",
    "cyan": "&H00FFFF00",
    "yellow": "&H0000FFFF",
    "pure_white": "&H00FFFFFF",
    "red_gold": "red_gold",
    "Red + Gold": "red_gold"
}

def clean_search_keyword(kw: str) -> str:
    """Strips out clipart/illustration boilerplate prefixes to get pristine photographic/video queries."""
    cleaned = re.sub(
        r"^(2d vector illustration of|minimalist icon graphic of|symbolic line art of|vector illustration of|minimalist icon of|illustration of|icon graphic of|icon of|graphic of|a photo of|photo of|picture of)\s+",
        "",
        kw.strip(),
        flags=re.IGNORECASE
    ).strip()
    return cleaned if cleaned else kw.strip()

def fetch_pexels_assets(
    edit_plan: dict,
    pexels_key: str = DEFAULT_PEXELS_KEY,
    download_dir: str = "assets/b_roll",
    default_asset_type: str = None
) -> dict:
    """Downloads pristine, high-res 1080x1920 Pexels stock photos or vertical video clips for visual cues."""
    os.makedirs(download_dir, exist_ok=True)
    cues = edit_plan.get("visual_cues") or edit_plan.get("b_roll_cues") or []
    headers = {"Authorization": pexels_key}

    for idx, cue in enumerate(cues):
        raw_keyword = cue.get("search_keyword", "abstract").strip()
        keyword = clean_search_keyword(raw_keyword)
        asset_type = cue.get("asset_type") or default_asset_type or "photo"
        slug = re.sub(r"[^\w]", "_", keyword.lower())[:25]
        
        words = keyword.split()
        search_terms = [keyword, " ".join(words[:3]) if len(words) > 3 else keyword, words[0]]

        # Video asset search
        if asset_type == "video":
            dest_path = os.path.join(download_dir, f"asset_{idx}_{slug}.mp4")
            if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 5000:
                downloaded = False
                for term in search_terms:
                    url = f"https://api.pexels.com/videos/search?query={term}&per_page=3&orientation=portrait"
                    try:
                        r = requests.get(url, headers=headers, timeout=10)
                        if r.status_code == 200:
                            videos = r.json().get("videos", [])
                            if videos:
                                v_files = videos[0].get("video_files", [])
                                portrait_files = [f for f in v_files if f.get("width", 1) <= f.get("height", 1)]
                                chosen = None
                                for cand in portrait_files:
                                    if cand.get("quality") in ["hd", "sd"]:
                                        chosen = cand
                                        break
                                if not chosen and portrait_files:
                                    chosen = portrait_files[0]
                                if not chosen and v_files:
                                    chosen = v_files[0]
                                if chosen and chosen.get("link"):
                                    v_data = requests.get(chosen["link"], timeout=25).content
                                    with open(dest_path, "wb") as f:
                                        f.write(v_data)
                                    downloaded = True
                                    break
                    except Exception as e:
                        print(f"Warning: Failed to fetch video for '{term}': {e}")
                if not downloaded:
                    # Fallback to photo if no video found
                    asset_type = "photo"

        # Photo asset search (Pristine 1080x1920 high resolution)
        if asset_type != "video":
            dest_path = os.path.join(download_dir, f"asset_{idx}_{slug}.jpg")
            if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1000:
                for term in search_terms:
                    url = f"https://api.pexels.com/v1/search?query={term}&per_page=3&orientation=portrait"
                    try:
                        r = requests.get(url, headers=headers, timeout=10)
                        if r.status_code == 200:
                            photos = r.json().get("photos", [])
                            if photos:
                                src = photos[0].get("src", {})
                                orig = src.get("original")
                                if orig:
                                    img_url = f"{orig}?auto=compress&cs=tinysrgb&fit=crop&w=1080&h=1920"
                                else:
                                    img_url = src.get("large2x") or src.get("large")
                                if img_url:
                                    img_data = requests.get(img_url, timeout=20).content
                                    with open(dest_path, "wb") as f:
                                        f.write(img_data)
                                    break
                    except Exception as e:
                        print(f"Warning: Failed to fetch photo for '{term}': {e}")

        cue["local_file"] = dest_path if os.path.exists(dest_path) else None

    return edit_plan

# Alias for backwards compatibility
fetch_pexels_photos = fetch_pexels_assets

def format_srt_time(seconds: float) -> str:
    """Formats seconds into strict SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def write_subtitles_srt(transcript_data: dict, srt_path: str = "subtitles.srt") -> str:
    """Converts transcript segments into single-line, collision-free SRT format."""
    segments = transcript_data.get("segments", [])
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments):
            start = format_srt_time(float(seg["start"]))
            end = format_srt_time(float(seg["end"]))
            text = seg["text"].strip()
            f.write(f"{idx + 1}\n{start} --> {end}\n{text}\n\n")
    return srt_path

def format_ass_time(seconds: float) -> str:
    """Formats seconds into ASS timestamp format H:MM:SS.cs"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"

NAME_INTRO_BLOCKLIST = {
    "hema", "hyma", "prasad", "dr", "doctor", "psychologist", "hymavathi",
    "name", "hello", "welcome", "myself", "i am", "this is", "speaking"
}

def format_callout_ass_text(text: str, color_mode: str = "white") -> str:
    """Formats center dynamic punch card text with ASS inline color tags (supporting white, red+gold, red, gold, cyan)."""
    words = text.strip().upper().split()
    if not words:
        return ""
    cm_lower = color_mode.strip().lower()
    if cm_lower in ["red_gold", "red + gold"]:
        if len(words) >= 2:
            first = rf"{{\c&H002020FF&}}{words[0]}"
            rest = " ".join([rf"{{\c&H0000E6FF&}}{w}" for w in words[1:]])
            return f"{first} {rest}"
        else:
            return rf"{{\c&H0000E6FF&}}{words[0]}"
    hex_code = ASS_COLOR_MAP.get(color_mode, ASS_COLOR_MAP.get(cm_lower, "&H00FFFFFF"))
    return rf"{{\c{hex_code}&}}{' '.join(words)}"

CALLOUT_STOP_WORDS = {
    "they", "must", "for", "giving", "given", "give", "learn", "your", "yours",
    "we", "our", "ours", "us", "you", "i", "me", "my", "he", "she", "it", "its",
    "the", "a", "an", "and", "or", "but", "so", "as", "at", "by", "in", "on", "to",
    "from", "with", "into", "onto", "over", "after", "before", "between", "through",
    "during", "above", "below", "of", "about", "against", "among", "can", "could",
    "will", "would", "shall", "should", "may", "might", "have", "has", "had", "having",
    "do", "does", "did", "doing", "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those", "what", "which", "who", "whom", "whose", "where",
    "when", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "than", "too", "very",
    "just", "now", "then", "like", "get", "getting", "got", "know", "knowing", "say",
    "saying", "said", "tell", "telling", "told", "make", "making", "made", "take", "taking",
    "them", "their", "theirs", "well", "much", "many", "even", "also", "still", "here",
    "let", "letting", "come", "coming", "go", "going", "see", "seeing", "saw", "seen",
    "stay", "staying", "stayed", "feel", "feeling", "felt", "think", "thinking", "thought"
}

def clean_callout_phrase(phrase: str) -> str:
    """Cleans a callout candidate, strictly stripping stop words, punctuation, and filler."""
    words = [w for w in re.sub(r"[^\w\s]", "", phrase).split() if len(w) > 1]
    # Filter out stop words and intro blocklist
    content_words = [w for w in words if w.lower() not in CALLOUT_STOP_WORDS and w.lower() not in NAME_INTRO_BLOCKLIST]
    if not content_words:
        return ""
    # Max 2 words
    return " ".join(content_words[:2]).upper()

def extract_curated_punch_callouts(edit_plan: dict, transcript_data: dict, highlight_words: str = "") -> list:
    """
    Extracts curated high-impact thematic punch callouts strictly filtering out filler words,
    stop words ('THEY MUST', 'FOR GIVING', 'LEARN YOUR'), and speaker introductions.
    Returns a list of dicts: [{'start': float, 'end': float, 'text': str, 'color': 'white', 'enabled': bool}]
    """
    callouts = []
    hl_list = [w.strip().lower() for w in highlight_words.split(",") if w.strip()]
    segments = transcript_data.get("segments", []) if transcript_data else []

    # 1. Custom user highlight keywords
    for seg in segments:
        text_raw = seg.get("text", "")
        text_lower = text_raw.lower()
        start_t = float(seg.get("start", 0))
        end_t = float(seg.get("end", 0))
        dur = end_t - start_t
        if dur < 0.4:
            continue

        for hw in hl_list:
            if hw in text_lower:
                clean_hw = clean_callout_phrase(hw)
                if clean_hw and len(clean_hw) >= 3:
                    callouts.append({
                        "start": round(start_t, 2),
                        "end": round(min(start_t + 2.0, end_t), 2),
                        "text": clean_hw,
                        "color": "white",
                        "enabled": True
                    })
                    break

    # 2. AI Editorial punch-ins
    punch_ins = edit_plan.get("punch_ins", []) if edit_plan else []
    for pi in punch_ins:
        pi_start = float(pi.get("start", 0))
        raw_candidate = pi.get("callout_text") or pi.get("keyword") or pi.get("reason", "")
        clean_cand = clean_callout_phrase(raw_candidate)
        
        # If candidate has valid content words
        if clean_cand and len(clean_cand) >= 3:
            if not any(abs(c["start"] - pi_start) < 2.0 for c in callouts):
                callouts.append({
                    "start": round(pi_start, 2),
                    "end": round(pi_start + 2.0, 2),
                    "text": clean_cand,
                    "color": "white",
                    "enabled": True
                })
                continue
                
        # Fallback: scan nearby segment for strong content nouns/verbs ONLY
        for seg in segments:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", 0))
            if abs(pi_start - seg_start) < 1.5:
                clean_seg_phrase = clean_callout_phrase(seg.get("text", ""))
                if clean_seg_phrase and len(clean_seg_phrase) >= 3:
                    if not any(abs(c["start"] - seg_start) < 2.0 for c in callouts):
                        callouts.append({
                            "start": round(seg_start, 2),
                            "end": round(min(seg_start + 2.0, seg_end), 2),
                            "text": clean_seg_phrase,
                            "color": "white",
                            "enabled": True
                        })
                break

    return callouts

def write_subtitles_ass(
    transcript_data: dict,
    ass_path: str = "subtitles.ass",
    video_w: int = 478,
    video_h: int = 850,
    font_name: str = "Montserrat Black",
    sub_font_size: int = None,
    callout_font_size: int = None,
    highlight_words: str = "",
    margin_v: int = 55,
    punch_callouts: list = None,
    callout_margin_v: int = 340,
    sub_color: str = "Pure White",
    sub_highlight_color: str = "Electric Gold",
    callout_default_color: str = "Pure White"
) -> str:
    """
    Generates an ASS subtitle file matching the viral reel format:
    - Font: Montserrat Black / League Spartan (ultra-bold heavy sans-serif)
    - All-caps pure white or custom colored text with thick black outline (Outline=3.8 to 4.8)
    - Zero background container box (BorderStyle=1)
    - Layer 0 (ReelSub): Lower desk dialogue subtitles with in-line gold keyword highlights
    - Layer 1 (PunchCallout): High-impact centered dynamic text cards
    """
    if sub_font_size is None:
        sub_font_size = max(17, int(video_w * 0.044) + 1) # ~22px (+1 point increase from 21px)
    if callout_font_size is None:
        callout_font_size = max(24, int(video_w * 0.078)) # ~37px

    sub_hex = ASS_COLOR_MAP.get(sub_color, ASS_COLOR_MAP.get(sub_color.strip().lower(), "&H00FFFFFF"))
    sub_hl_hex = ASS_COLOR_MAP.get(sub_highlight_color, ASS_COLOR_MAP.get(sub_highlight_color.strip().lower(), "&H0000E6FF"))
    callout_hex = ASS_COLOR_MAP.get(callout_default_color, ASS_COLOR_MAP.get(callout_default_color.strip().lower(), "&H00FFFFFF"))
    
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ReelSub,{font_name},{sub_font_size},{sub_hex},&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3.8,1.4,2,20,20,{margin_v},1
Style: PunchCallout,{font_name},{callout_font_size},{callout_hex},&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,4.8,2.0,2,20,20,{callout_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Parse highlight keywords
    hl_list = [w.strip() for w in highlight_words.split(",") if w.strip()]

    events = []
    # Layer 0: Dialogue Subtitles at lower desk
    if transcript_data:
        for seg in transcript_data.get("segments", []):
            start_sec = float(seg["start"])
            end_sec = float(seg["end"])
            start_str = format_ass_time(start_sec)
            end_str = format_ass_time(end_sec)
            text = seg["text"].strip().upper()
            
            # Apply in-line keyword highlighting with ASS color tags
            for hw in hl_list:
                pattern = re.compile(rf"\b({re.escape(hw.upper())})\b", re.IGNORECASE)
                text = pattern.sub(rf"{{\\c{sub_hl_hex}&}}\1{{\\c{sub_hex}&}}", text)

            events.append(f"Dialogue: 0,{start_str},{end_str},ReelSub,,0,0,0,,{text}")

    # Layer 1: Center Dynamic Text Cards (Outlined typography)
    if punch_callouts:
        for callout in punch_callouts:
            if not callout.get("enabled", True):
                continue
            raw_callout_txt = callout.get("text", "").strip()
            if not raw_callout_txt:
                continue
            s_str = format_ass_time(float(callout["start"]))
            e_str = format_ass_time(float(callout["end"]))
            c_color = callout.get("color", callout_default_color)
            styled_txt = format_callout_ass_text(raw_callout_txt, c_color)
            events.append(f"Dialogue: 1,{s_str},{e_str},PunchCallout,,0,0,0,,{styled_txt}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
        
    return ass_path

def burn_subtitles_ffmpeg(
    video_input_path: str,
    sub_path: str,
    video_output_path: str,
    margin_v: int = 55,
    video_w: int = 478,
    video_h: int = 850
) -> str:
    """Fast subtitle burning using FFmpeg directly with Montserrat Black bundled font (zero container box, sharp outline)."""
    ffmpeg = get_ffmpeg_exe()
    safe_sub = sub_path.replace("\\", "/")
    fonts_dir = "assets/fonts"
    
    if sub_path.lower().endswith(".ass"):
        vf_arg = f"subtitles='{safe_sub}':fontsdir='{fonts_dir}'"
    else:
        font_size = max(16, int(video_w * 0.044))
        style_opts = (
            f"PlayResX={video_w},"
            f"PlayResY={video_h},"
            "FontName=Montserrat Black,"
            f"FontSize={font_size},"
            "Bold=1,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "BorderStyle=1,"
            "Outline=3.8,"
            "Shadow=1.4,"
            "Alignment=2,"
            f"MarginV={margin_v}"
        )
        vf_arg = f"subtitles='{safe_sub}':fontsdir='{fonts_dir}':force_style='{style_opts}'"

    cmd = [
        ffmpeg, "-y", "-i", video_input_path,
        "-vf", vf_arg,
        "-c:a", "copy", video_output_path
    ]
    subprocess.run(cmd, check=True)
    return video_output_path


from PIL import Image, ImageDraw, ImageFilter, ImageFont

def create_card_pil(img_path, target_w=340, target_h=190, radius=18):
    """Creates a sleek, high-production photo card with rounded corners, white border, and soft drop shadow."""
    im = Image.open(img_path).convert('RGBA')
    aspect_target = target_w / target_h
    aspect_im = im.width / im.height
    if aspect_im > aspect_target:
        new_w = int(im.height * aspect_target)
        left = (im.width - new_w) // 2
        im = im.crop((left, 0, left + new_w, im.height))
    else:
        new_h = int(im.width / aspect_target)
        top = (im.height - new_h) // 2
        im = im.crop((0, top, im.width, top + new_h))
    im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    mask = Image.new('L', (target_w, target_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (target_w, target_h)], radius=radius, fill=255)
    
    pad = 16
    total_w = target_w + pad * 2
    total_h = target_h + pad * 2
    canvas = Image.new('RGBA', (total_w, total_h), (0, 0, 0, 0))
    
    # Soft drop shadow
    shadow = Image.new('RGBA', (target_w + 4, target_h + 4), (0, 0, 0, 160))
    s_mask = Image.new('L', (target_w + 4, target_h + 4), 0)
    s_draw = ImageDraw.Draw(s_mask)
    s_draw.rounded_rectangle([(0, 0), (target_w + 4, target_h + 4)], radius=radius + 2, fill=255)
    
    shadow_layer = Image.new('RGBA', (total_w, total_h), (0, 0, 0, 0))
    shadow_layer.paste(shadow, (pad - 2, pad + 4), s_mask)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(8))
    canvas.paste(shadow_layer, (0, 0), shadow_layer)
    
    # Clean crisp white border
    border_w = target_w + 6
    border_h = target_h + 6
    border_img = Image.new('RGBA', (border_w, border_h), (255, 255, 255, 245))
    b_mask = Image.new('L', (border_w, border_h), 0)
    b_draw = ImageDraw.Draw(b_mask)
    b_draw.rounded_rectangle([(0, 0), (border_w, border_h)], radius=radius + 3, fill=255)
    
    canvas.paste(border_img, (pad - 3, pad - 3), b_mask)
    canvas.paste(im, (pad, pad), mask)
    return canvas

def make_animated_card_clip(img_path, duration, video_w, video_h, card_y_pct: float = 0.14):
    """Builds a MoviePy clip with pop-in zoom entry, Ken Burns drift, positioned in the upper safe zone above subject's head."""
    target_w = int(video_w * 0.70)
    target_h = int(target_w * 0.5625)
    y_center = int(video_h * card_y_pct) # 0.14 moves pictures up by 3 points, well clear of subject's face/hair
    
    card_pil = create_card_pil(img_path, target_w, target_h)
    base_w, base_h = card_pil.size
    card_np = np.array(card_pil)
    card_clip = ImageClip(card_np, ismask=False).set_duration(duration)
    
    def scale_fn(t):
        if t < 0.30:
            p = t / 0.30
            ease = 1.0 - (1.0 - p)**3
            return 0.70 + 0.30 * ease
        else:
            return 1.0 + 0.04 * (t / max(0.1, duration))
            
    def pos_fn(t):
        s = scale_fn(t)
        cw = base_w * s
        ch = base_h * s
        return ((video_w - cw) / 2, y_center - ch / 2)
        
    animated = card_clip.resize(scale_fn).set_position(pos_fn)
    try:
        animated = animated.crossfadein(0.20).crossfadeout(0.25)
    except Exception:
        pass
    return animated

def create_reel_kinetic_word_pil(text: str, video_w: int = 478) -> Image.Image:
    """
    Renders pure, broadcast-grade kinetic typography identical to viral podcast reels (Sudheer Talks style).
    - NO enclosing pill container or button box.
    - Heavy Impact/Arial Black font.
    - Dual-tone color hierarchy (Crimson Red + Electric Gold / Gold / Red).
    - Deep drop shadow + black outer stroke + alpha transparency.
    """
    clean_text = text.strip().upper()
    font_size = max(24, int(video_w * 0.095)) # ~45px on 478w
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/impact.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/ariblk.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    words = clean_text.split()
    dummy = Image.new("RGBA", (1, 1))
    draw_d = ImageDraw.Draw(dummy)

    # Word colors: first word crimson red, subsequent words electric gold
    word_colors = []
    if len(words) >= 2:
        word_colors = [(255, 38, 38)] + [(255, 230, 0)] * (len(words) - 1)
    else:
        word_colors = [(255, 230, 0)]

    space_w = draw_d.textbbox((0, 0), " ", font=font)[2]
    total_w = 0
    max_h = 0
    word_metrics = []
    for w in words:
        bbox = draw_d.textbbox((0, 0), w, font=font)
        ww = bbox[2] - bbox[0]
        wh = bbox[3] - bbox[1]
        word_metrics.append((w, bbox, ww, wh))
        total_w += ww
        max_h = max(max_h, wh)
    total_w += space_w * max(0, len(words) - 1)

    pad = 28
    canvas_w = total_w + pad * 2
    canvas_h = max_h + pad * 2

    # 1. Soft deep drop shadow layer
    shadow_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow_img)
    curr_x = pad
    for w, bbox, ww, wh in word_metrics:
        s_draw.text((curr_x - bbox[0], pad - bbox[1] + 5), w, font=font, fill=(0, 0, 0, 220))
        curr_x += ww + space_w
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(5))

    # 2. Crisp stroke and vibrant text fill layer
    text_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    t_draw = ImageDraw.Draw(text_img)
    curr_x = pad
    for idx, (w, bbox, ww, wh) in enumerate(word_metrics):
        col = word_colors[idx]
        t_draw.text(
            (curr_x - bbox[0], pad - bbox[1]),
            w,
            font=font,
            fill=col,
            stroke_width=4,
            stroke_fill=(0, 0, 0, 255)
        )
        curr_x += ww + space_w

    return Image.alpha_composite(shadow_img, text_img)

def make_reel_kinetic_word_clip(text: str, start: float, duration: float, video_w: int, video_h: int, y_pos: int = None):
    """Generates an animated kinetic text clip with punch-in zoom entry and clean fade-out."""
    if y_pos is None:
        y_pos = int(video_h * 0.58) # chest area, completely clear of face and bottom desk
        
    text_pil = create_reel_kinetic_word_pil(text, video_w)
    base_w, base_h = text_pil.size
    text_np = np.array(text_pil)
    
    clip = ImageClip(text_np, ismask=False).set_duration(duration).set_start(start)
    
    def scale_fn(t):
        if t < 0.18:
            p = t / 0.18
            return 0.88 + 0.12 * (1.0 - (1.0 - p)**3)
        elif t > duration - 0.20:
            p = (duration - t) / 0.20
            return max(0.8, p)
        else:
            return 1.0
            
    def pos_fn(t):
        s = scale_fn(t)
        cw = base_w * s
        ch = base_h * s
        return ((video_w - cw) / 2, y_pos - ch / 2)
        
    animated = clip.resize(scale_fn).set_position(pos_fn)
    try:
        animated = animated.crossfadein(0.08).crossfadeout(0.18)
    except Exception:
        pass
    return animated

def make_fullscreen_broll_clip(asset_path: str, duration: float, video_w: int, video_h: int):
    """
    Builds a full-screen vertical 9:16 motion B-roll clip.
    - If image: crops & scales oversized by 12% to prevent edge clipping, applies smooth Ken Burns zoom,
      and soft alpha crossfades directly over the main video (no black background flash).
    - If video: loops/trims to duration, center crops to 9:16, strips audio, and adds soft crossfades.
    """
    is_image = asset_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    if is_image:
        im = Image.open(asset_path).convert('RGB')
        oversize_w = int(video_w * 1.12)
        oversize_h = int(video_h * 1.12)
        
        target_aspect = video_w / video_h
        im_aspect = im.width / im.height
        if im_aspect > target_aspect:
            new_w = int(im.height * target_aspect)
            left = (im.width - new_w) // 2
            im = im.crop((left, 0, left + new_w, im.height))
        else:
            new_h = int(im.width / target_aspect)
            top = (im.height - new_h) // 2
            im = im.crop((0, top, im.width, top + new_h))
        im = im.resize((oversize_w, oversize_h), Image.Resampling.LANCZOS)
        
        base_np = np.array(im)
        clip = ImageClip(base_np).set_duration(duration)
        
        def zoom_scale(t):
            return 1.0 + 0.05 * (t / max(0.1, duration))
            
        def center_pos(t):
            s = zoom_scale(t)
            cur_w = oversize_w * s
            cur_h = oversize_h * s
            return ((video_w - cur_w) / 2, (video_h - cur_h) / 2)
            
        animated = clip.resize(zoom_scale).set_position(center_pos)
        try:
            animated = animated.crossfadein(0.22).crossfadeout(0.25)
        except Exception:
            pass
        return animated
    else:
        # Video asset
        v_clip = VideoFileClip(asset_path).without_audio()
        if v_clip.duration is not None and v_clip.duration < duration:
            v_clip = v_clip.loop(duration=duration)
        else:
            v_clip = v_clip.subclip(0, duration)
            
        # Fit to 9:16
        clip_aspect = v_clip.w / v_clip.h
        target_aspect = video_w / video_h
        if clip_aspect > target_aspect:
            v_clip = v_clip.resize(height=video_h)
        else:
            v_clip = v_clip.resize(width=video_w)
            
        v_clip = crop(v_clip, x_center=v_clip.w / 2, y_center=v_clip.h / 2, width=video_w, height=video_h)
        try:
            v_clip = v_clip.crossfadein(0.22).crossfadeout(0.25)
        except Exception:
            pass
        return v_clip

def assemble_final_video(
    raw_video_path: str,
    edit_plan: dict,
    sub_path: str,
    output_path: str = "final_edited_video.mp4",
    transcript_data: dict = None,
    highlight_words: str = "",
    sub_margin_v: int = 55,
    card_y_pct: float = 0.14,
    punch_callouts: list = None,
    font_name: str = "Montserrat Black",
    visual_display_mode: str = "fullscreen"
) -> str:
    """Composites raw video with full-screen motion B-roll cuts or upper photo cards, and burns ASS subtitles + kinetic callouts."""
    main_clip = VideoFileClip(raw_video_path)
    combined = main_clip

    visual_cues = edit_plan.get("visual_cues") or edit_plan.get("b_roll_cues") or []
    overlays = [combined]

    # Layer Visual Cues (Full-Screen Cutaways or Upper Photo Cards)
    for idx, cue in enumerate(visual_cues):
        if not cue.get("enabled", True):
            continue
        local_path = cue.get("local_file")
        start = float(cue.get("start", 0))
        end = float(cue.get("end", 0))
        duration = end - start

        if local_path and os.path.exists(local_path) and duration > 0:
            is_image = local_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            cue_mode = cue.get("display_mode") or visual_display_mode
            if cue_mode in ["fullscreen", "Full-Screen Cutaway (Recommended)", "cutaway"]:
                print(f"Overlaying Full-Screen Motion B-Roll '{cue.get('search_keyword')}' at {start:.2f}s - {(start + duration):.2f}s...")
                fs_clip = make_fullscreen_broll_clip(local_path, duration, combined.w, combined.h)
                fs_clip = fs_clip.set_start(start)
                overlays.append(fs_clip)
            else: # "card" / "Upper Floating Card"
                print(f"Overlaying Upper Photo Card '{cue.get('search_keyword')}' at {start:.2f}s - {(start + duration):.2f}s...")
                if is_image:
                    card_clip = make_animated_card_clip(local_path, duration, combined.w, combined.h, card_y_pct=card_y_pct)
                    card_clip = card_clip.set_start(start)
                    overlays.append(card_clip)
                else:
                    fs_clip = make_fullscreen_broll_clip(local_path, duration, combined.w, combined.h)
                    fs_clip = fs_clip.set_start(start)
                    overlays.append(fs_clip)

    final_render = CompositeVideoClip(overlays)
    temp_output = "temp_assembled.mp4"
    final_render.write_videofile(
        temp_output,
        fps=main_clip.fps,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=4,
        remove_temp=False
    )
    final_render.close()
    main_clip.close()

    # Burn subtitles and kinetic punch callouts with FFmpeg (fast ~1.5s, no video re-encoding)
    burn_subtitles_ffmpeg(
        video_input_path=temp_output,
        sub_path=sub_path,
        video_output_path=output_path,
        margin_v=sub_margin_v,
        video_w=combined.w,
        video_h=combined.h
    )

    # Cleanup temp sound files, keep temp_assembled.mp4 for instant subtitle/callout re-burns
    for temp_f in ["temp_assembledTEMP_MPY_wvf_snd.mp4"]:
        if os.path.exists(temp_f):
            try:
                os.remove(temp_f)
            except Exception:
                pass

    return output_path

def run_pipeline(
    raw_video_path: str,
    output_path: str = "final_edited_video.mp4",
    gemini_key: str = DEFAULT_GEMINI_KEY,
    pexels_key: str = DEFAULT_PEXELS_KEY,
    vocab_hints: str = "",
    highlight_words: str = "",
    sub_margin_v: int = 55,
    sub_font_size: int = None,
    card_y_pct: float = 0.14,
    punch_callouts: list = None,
    font_name: str = "Montserrat Black",
    visual_display_mode: str = "fullscreen",
    sub_color: str = "Pure White",
    sub_highlight_color: str = "Electric Gold",
    callout_color: str = "Pure White",
    progress_callback = None
):
    """End-to-end processing pipeline with progress updates."""
    def notify(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)
        print(f"[{pct}%] {msg}")

    # Stage 1: Audio Extraction & Transcription (0-30%)
    notify(5, "Extracting audio track from video...")
    audio_path = extract_audio(raw_video_path, "temp_audio.wav")
    
    notify(15, "Transcribing and translating audio with Gemini...")
    transcript_data = transcribe_audio_gemini(audio_path, api_key=gemini_key, vocab_hints=vocab_hints)
    with open("transcript_english.json", "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2)

    # Stage 2: AI Editorial Analysis (30-55%)
    notify(35, "AI Director Agent analyzing video narrative...")
    edit_plan = generate_edit_plan_gemini(transcript_data, api_key=gemini_key)
    with open("edit_plan.json", "w", encoding="utf-8") as f:
        json.dump(edit_plan, f, indent=2)

    # Stage 3: Visual Stock Asset Retrieval (55-75%)
    notify(60, "Retrieving high-resolution contextual photos/videos from Pexels...")
    updated_plan = fetch_pexels_assets(edit_plan, pexels_key=pexels_key)
    with open("edit_plan.json", "w", encoding="utf-8") as f:
        json.dump(updated_plan, f, indent=2)

    # Stage 4: Subtitles & Video Assembly (75-100%)
    notify(75, f"Generating synchronized {font_name} ASS subtitles & callouts...")
    cues = updated_plan.get("visual_cues") or updated_plan.get("b_roll_cues") or []
    
    # Extract curated punch callouts (strictly filtering out names and speaker intros)
    if punch_callouts is None:
        punch_callouts = extract_curated_punch_callouts(updated_plan, transcript_data, highlight_words)

    # Get video dimensions
    probe_clip = VideoFileClip(raw_video_path)
    vid_w, vid_h = probe_clip.w, probe_clip.h
    probe_clip.close()
    
    ass_path = write_subtitles_ass(
        transcript_data,
        ass_path="subtitles.ass",
        video_w=vid_w,
        video_h=vid_h,
        font_name=font_name,
        sub_font_size=sub_font_size,
        highlight_words=highlight_words,
        margin_v=sub_margin_v,
        punch_callouts=punch_callouts,
        sub_color=sub_color,
        sub_highlight_color=sub_highlight_color,
        callout_default_color=callout_color
    )
    # Also write standard srt for fallback
    write_subtitles_srt(transcript_data, "subtitles.srt")

    notify(85, f"Rendering {visual_display_mode} motion B-roll, {font_name} reel typography & subtitles...")
    final_video = assemble_final_video(
        raw_video_path,
        updated_plan,
        ass_path,
        output_path=output_path,
        transcript_data=transcript_data,
        highlight_words=highlight_words,
        sub_margin_v=sub_margin_v,
        card_y_pct=card_y_pct,
        punch_callouts=punch_callouts,
        font_name=font_name,
        visual_display_mode=visual_display_mode
    )

    notify(100, "Video editing complete!")
    return final_video, updated_plan, transcript_data, punch_callouts


