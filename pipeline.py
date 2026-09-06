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
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
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
    """AI Director Agent: Analyzes transcript to select high-relevance visual cues and punch-ins with auto-fallback."""
    genai.configure(api_key=api_key, transport="rest")

    prompt = """
    You are an award-winning social video editor creating viral, engaging short-form videos (Reels, TikTok, Shorts, YouTube).
    Analyze the provided timestamped transcript and return a strict JSON editing blueprint.
    
    YOUR EDITING STRATEGY:
    1. HOOK DETECTION:
       - Identify if there is a compelling, high-energy sentence or curiosity gap within the video that should serve as an upfront teaser hook (3 to 6 seconds max).
       - If the video already starts with a strong, natural opening, set "hook_segment": null.
    2. VISUAL ASSETS (Pictures & B-Roll Cues):
       - Identify 4 to 7 key moments where a visual aid enhances comprehension or retention.
       - DO NOT restrict yourself to literal physical nouns. Include conceptual, metaphorical, or emotional visuals (e.g., "market crash graph", "frustrated developer", "serene morning nature", "cryptocurrency wallet", "celebration confetti").
       - Always output English keywords for "search_keyword" (optimized for stock photo/video APIs).
       - Keep each visual appearance between 3.5 and 4.5 seconds. Space them out naturally across the timeline.
    3. PUNCH-INS (Dynamic Concepts):
       - Identify 2 to 4 high-emphasis words, surprising statistics, or conceptual punchlines to emphasize (1 to 2 seconds).
       - CRITICAL RULE FOR PUNCH WORDS: NEVER select speaker names, greetings, or self-introductions (e.g., 'Dr. Hema', 'I am...', 'My name is...', 'Psychologist'). Punch words must ONLY be punchy, thematic high-impact concepts (e.g., 'POOR COMMUNICATION', 'FEAR OF ENGLISH', 'REJECTED', 'JOB INTERVIEW', 'CONFIDENCE BLOCK').
    
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
        {"start": float, "end": float, "reason": "High-impact thematic concept (never speaker names/intros)"}
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
            print(f"Editorial plan generated successfully using model '{model_name}'")
            return data
        except Exception as e:
            last_err = e
            print(f"Model '{model_name}' planning error: {e}. Falling back to next candidate...")
            continue
    raise last_err

def fetch_pexels_photos(edit_plan: dict, pexels_key: str = DEFAULT_PEXELS_KEY, download_dir: str = "assets/b_roll") -> dict:
    """Downloads high-res Pexels stock photos for all visual cues in the plan."""
    os.makedirs(download_dir, exist_ok=True)
    cues = edit_plan.get("visual_cues") or edit_plan.get("b_roll_cues") or []
    headers = {"Authorization": pexels_key}

    for idx, cue in enumerate(cues):
        keyword = cue.get("search_keyword", "abstract")
        slug = re.sub(r"[^\w]", "_", keyword.strip().lower())[:25]
        dest_path = os.path.join(download_dir, f"asset_{idx}_{slug}.jpg")

        if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1000:
            words = keyword.split()
            search_terms = [keyword, " ".join(words[:3]) if len(words) > 3 else keyword, words[0]]
            for term in search_terms:
                url = f"https://api.pexels.com/v1/search?query={term}&per_page=1"
                try:
                    r = requests.get(url, headers=headers, timeout=10)
                    if r.status_code == 200:
                        photos = r.json().get("photos", [])
                        if photos:
                            src = photos[0].get("src", {})
                            img_url = src.get("large2x") or src.get("portrait") or src.get("large") or src.get("original")
                            if img_url:
                                img_data = requests.get(img_url, timeout=15).content
                                with open(dest_path, "wb") as f:
                                    f.write(img_data)
                                break
                except Exception as e:
                    print(f"Warning: Failed to fetch photo for '{term}': {e}")

        cue["local_file"] = dest_path if os.path.exists(dest_path) else None

    return edit_plan

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

def format_callout_ass_text(text: str, color_mode: str = "red_gold") -> str:
    """Formats center punch text with ASS inline color tags matching viral podcast reels."""
    words = text.strip().upper().split()
    if not words:
        return ""
    if color_mode == "red_gold":
        if len(words) >= 2:
            first = rf"{{\c&H002020FF&}}{words[0]}"
            rest = " ".join([rf"{{\c&H0000E6FF&}}{w}" for w in words[1:]])
            return f"{first} {rest}"
        else:
            return rf"{{\c&H0000E6FF&}}{words[0]}"
    elif color_mode == "red":
        return rf"{{\c&H002020FF&}}{' '.join(words)}"
    elif color_mode == "gold":
        return rf"{{\c&H0000E6FF&}}{' '.join(words)}"
    else: # white
        return rf"{{\c&H00FFFFFF&}}{' '.join(words)}"

def extract_curated_punch_callouts(edit_plan: dict, transcript_data: dict, highlight_words: str = "") -> list:
    """
    Extracts high-impact thematic punch callouts while strictly filtering out names and speaker introductions.
    Returns a list of dicts: [{'start': float, 'end': float, 'text': str, 'color': 'red_gold', 'enabled': bool}]
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
                callouts.append({
                    "start": round(start_t, 2),
                    "end": round(min(start_t + 2.2, end_t), 2),
                    "text": hw.upper(),
                    "color": "red_gold",
                    "enabled": True
                })
                break

    # 2. AI Editorial punch-ins (strictly filtering speaker names/intros)
    punch_ins = edit_plan.get("punch_ins", []) if edit_plan else []
    for pi in punch_ins:
        pi_start = float(pi.get("start", 0))
        for seg in segments:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", 0))
            if abs(pi_start - seg_start) < 1.5:
                clean_words = [cw for cw in re.sub(r"[^\w\s]", "", seg.get("text", "")).split() if len(cw) > 2]
                filtered = [w for w in clean_words if w.lower() not in NAME_INTRO_BLOCKLIST]
                if filtered:
                    phrase = " ".join(filtered[:2]).upper()
                    if not any(abs(c["start"] - seg_start) < 2.0 for c in callouts):
                        callouts.append({
                            "start": round(seg_start, 2),
                            "end": round(min(seg_start + 2.2, seg_end), 2),
                            "text": phrase,
                            "color": "red_gold",
                            "enabled": True
                        })
                break

    return callouts

def write_subtitles_ass(
    transcript_data: dict,
    ass_path: str = "subtitles.ass",
    video_w: int = 478,
    video_h: int = 850,
    highlight_words: str = "",
    margin_v: int = 55,
    punch_callouts: list = None,
    callout_margin_v: int = 330
) -> str:
    """
    Generates an ASS subtitle file with two professional layers:
    - Layer 0 (ReelSub): Lower desk dialogue subtitles with in-line gold word highlights.
    - Layer 1 (PunchCallout): High-impact center kinetic typography in distressed Impact font.
    """
    font_size = max(16, int(video_w * 0.044)) # ~21px
    callout_font_size = max(24, int(video_w * 0.088)) # ~42px
    
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ReelSub,Arial Black,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3.2,1.2,2,20,20,{margin_v},1
Style: PunchCallout,Impact,{callout_font_size},&H0000E6FF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,4.5,2.5,2,20,20,{callout_margin_v},1

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
            
            # Apply in-line keyword highlighting with ASS color tags (vibrant electric gold)
            for hw in hl_list:
                pattern = re.compile(rf"\b({re.escape(hw.upper())})\b", re.IGNORECASE)
                text = pattern.sub(r"{\\c&H0000E6FF&}\1{\\c&H00FFFFFF&}", text)

            events.append(f"Dialogue: 0,{start_str},{end_str},ReelSub,,0,0,0,,{text}")

    # Layer 1: Center Punch Callouts (Pure Kinetic Typography in safe chest area)
    if punch_callouts:
        for callout in punch_callouts:
            if not callout.get("enabled", True):
                continue
            raw_callout_txt = callout.get("text", "").strip()
            if not raw_callout_txt:
                continue
            s_str = format_ass_time(float(callout["start"]))
            e_str = format_ass_time(float(callout["end"]))
            styled_txt = format_callout_ass_text(raw_callout_txt, callout.get("color", "red_gold"))
            events.append(f"Dialogue: 1,{s_str},{e_str},PunchCallout,,0,0,0,,{styled_txt}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
        
    return ass_path

def burn_subtitles_ffmpeg(
    video_input_path: str,
    sub_path: str,
    video_output_path: str,
    margin_v: int = 320,
    video_w: int = 478,
    video_h: int = 850
) -> str:
    """Fast subtitle burning using FFmpeg directly (takes ~1-2 seconds, zero video re-encoding required)."""
    ffmpeg = get_ffmpeg_exe()
    safe_sub = sub_path.replace("\\", "/")
    
    if sub_path.lower().endswith(".ass"):
        vf_arg = f"subtitles='{safe_sub}'"
    else:
        font_size = max(16, int(video_w * 0.044))
        style_opts = (
            f"PlayResX={video_w},"
            f"PlayResY={video_h},"
            "FontName=Arial Black,"
            f"FontSize={font_size},"
            "Bold=1,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "BorderStyle=1,"
            "Outline=3,"
            "Shadow=1.5,"
            "Alignment=2,"
            f"MarginV={margin_v}"
        )
        vf_arg = f"subtitles='{safe_sub}':force_style='{style_opts}'"

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

def make_animated_card_clip(img_path, duration, video_w, video_h):
    """Builds a MoviePy clip with pop-in zoom entry, Ken Burns drift, in upper third safe zone."""
    target_w = int(video_w * 0.72)
    target_h = int(target_w * 0.5625)
    y_center = int(video_h * 0.17) # Upper third safe zone above head
    
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

def assemble_final_video(
    raw_video_path: str,
    edit_plan: dict,
    sub_path: str,
    output_path: str = "final_edited_video.mp4",
    transcript_data: dict = None,
    highlight_words: str = "",
    sub_margin_v: int = 55,
    punch_callouts: list = None
) -> str:
    """Composites raw video with upper animated photo cards, and burns ASS subtitles + kinetic callouts."""
    main_clip = VideoFileClip(raw_video_path)
    combined = main_clip

    visual_cues = edit_plan.get("visual_cues") or edit_plan.get("b_roll_cues") or []
    overlays = [combined]

    # Layer Upper Photo Cards (Exclusively in upper third safe zone above head)
    for idx, cue in enumerate(visual_cues):
        local_path = cue.get("local_file")
        start = float(cue.get("start", 0))
        end = float(cue.get("end", 0))
        duration = end - start

        if local_path and os.path.exists(local_path) and duration > 0:
            is_image = local_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            print(f"Overlaying Upper Photo Card '{cue.get('search_keyword')}' at {start:.2f}s - {(start + duration):.2f}s...")
            
            if is_image:
                card_clip = make_animated_card_clip(local_path, duration, combined.w, combined.h)
                card_clip = card_clip.set_start(start)
                overlays.append(card_clip)
            else:
                asset_clip = VideoFileClip(local_path).without_audio()
                if asset_clip.duration is not None and asset_clip.duration < duration:
                    asset_clip = asset_clip.loop(duration=duration)
                else:
                    asset_clip = asset_clip.subclip(0, duration)
                asset_clip = asset_clip.resize(height=combined.h)
                if asset_clip.w < combined.w:
                    asset_clip = asset_clip.resize(width=combined.w)
                asset_clip = crop(asset_clip, x_center=asset_clip.w / 2, y_center=asset_clip.h / 2,
                                  width=combined.w, height=combined.h)
                asset_clip = asset_clip.set_position("center").set_start(start)
                overlays.append(asset_clip)

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
    punch_callouts: list = None,
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
    
    notify(15, "Transcribing and translating audio with Gemini 3.7 Flash...")
    transcript_data = transcribe_audio_gemini(audio_path, api_key=gemini_key, vocab_hints=vocab_hints)
    with open("transcript_english.json", "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2)

    # Stage 2: AI Editorial Analysis (30-55%)
    notify(35, "AI Director Agent analyzing video narrative...")
    edit_plan = generate_edit_plan_gemini(transcript_data, api_key=gemini_key)
    with open("edit_plan.json", "w", encoding="utf-8") as f:
        json.dump(edit_plan, f, indent=2)

    # Stage 3: Visual Stock Asset Retrieval (55-75%)
    notify(60, "Retrieving high-resolution contextual photos from Pexels...")
    updated_plan = fetch_pexels_photos(edit_plan, pexels_key=pexels_key)
    with open("edit_plan.json", "w", encoding="utf-8") as f:
        json.dump(updated_plan, f, indent=2)

    # Stage 4: Subtitles & Video Assembly (75-100%)
    notify(75, "Generating synchronized ASS subtitles with in-line gold word highlights...")
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
        highlight_words=highlight_words,
        margin_v=sub_margin_v,
        punch_callouts=punch_callouts
    )
    # Also write standard srt for fallback
    write_subtitles_srt(transcript_data, "subtitles.srt")

    notify(85, "Rendering upper photo cards, kinetic reel typography & subtitles...")
    final_video = assemble_final_video(
        raw_video_path,
        updated_plan,
        ass_path,
        output_path=output_path,
        transcript_data=transcript_data,
        highlight_words=highlight_words,
        sub_margin_v=sub_margin_v,
        punch_callouts=punch_callouts
    )

    notify(100, "Video editing complete!")
    return final_video, updated_plan, transcript_data, punch_callouts

