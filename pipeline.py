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

def transcribe_audio_gemini(audio_path: str, api_key: str = DEFAULT_GEMINI_KEY) -> dict:
    """Uses Gemini 3.6 Flash to transcribe and translate audio into concise English subtitle segments."""
    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel("gemini-3.6-flash", generation_config={"response_mime_type": "application/json"})
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    prompt = """
    Listen to this entire audio track carefully.
    Transcribe and translate the entire speech into natural, conversational English subtitle segments.
    
    CRITICAL GUIDELINES FOR HIGH-ENGAGEMENT SOCIAL CAPTIONS:
    - Keep each segment short and punchy: strictly 3 to 6 words (under 30 characters).
    - Accurately timestamp start and end in seconds (e.g., 0.8 to 4.2).
    - Cover the entire speech timeline from beginning to end without gaps or skipping.
    
    Return strict JSON matching this schema:
    {
      "language": "detected_code",
      "segments": [
        {"id": 1, "start": float, "end": float, "text": "Short punchy phrase"}
      ]
    }
    """
    response = model.generate_content([{"mime_type": "audio/wav", "data": audio_bytes}, prompt])
    clean_json = response.text.strip().removeprefix("```json").removesuffix("```").strip()
    data = json.loads(clean_json)
    if isinstance(data, list):
        data = {"segments": data}
    return data

def generate_edit_plan_gemini(transcript_data: dict, api_key: str = DEFAULT_GEMINI_KEY) -> dict:
    """AI Director Agent: Analyzes transcript to select high-relevance visual cues and punch-ins."""
    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel("gemini-3.6-flash", generation_config={"response_mime_type": "application/json"})

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
    response = model.generate_content([prompt, json.dumps(transcript_data)])
    clean_json = response.text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(clean_json)

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

from PIL import Image, ImageDraw, ImageFilter

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
    """Builds a MoviePy clip with pop-in zoom entry, Ken Burns drift, and safe upper framing."""
    target_w = int(video_w * 0.72)
    target_h = int(target_w * 0.5625)
    y_center = int(video_h * 0.17)
    
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

def assemble_final_video(
    raw_video_path: str,
    edit_plan: dict,
    srt_path: str,
    output_path: str = "final_edited_video.mp4",
    margin_v: int = 15
) -> str:
    """Composites raw video with animated photo cards and burns modern desk-aligned subtitles."""
    main_clip = VideoFileClip(raw_video_path)
    combined = main_clip

    visual_cues = edit_plan.get("visual_cues") or edit_plan.get("b_roll_cues") or []
    overlays = [combined]

    for cue in visual_cues:
        local_path = cue.get("local_file")
        start = float(cue.get("start", 0))
        end = float(cue.get("end", 0))
        duration = end - start

        if local_path and os.path.exists(local_path) and duration > 0:
            is_image = local_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
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

    # Burn subtitles with FFmpeg
    ffmpeg = get_ffmpeg_exe()
    style_opts = (
        "FontName=Arial,"
        "FontSize=12,"
        "Bold=1,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H90000000,"
        "BorderStyle=3,"
        "Outline=2,"
        "Shadow=0,"
        "Alignment=2,"
        f"MarginV={margin_v}"
    )
    safe_srt = srt_path.replace("\\", "/")
    cmd = [
        ffmpeg, "-y", "-i", temp_output,
        "-vf", f"subtitles='{safe_srt}':force_style='{style_opts}'",
        "-c:a", "copy", output_path
    ]
    subprocess.run(cmd, check=True)

    # Cleanup temp
    for temp_f in [temp_output, "temp_assembledTEMP_MPY_wvf_snd.mp4"]:
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
    translucent_opacity: float = 0.30,
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
    
    notify(15, "Transcribing and translating audio with Gemini 3.6 Flash...")
    transcript_data = transcribe_audio_gemini(audio_path, api_key=gemini_key)
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
    notify(75, "Generating synchronized single-line subtitle track...")
    srt_path = write_subtitles_srt(transcript_data, "subtitles.srt")

    notify(85, "Rendering animated photo cards and burning subtitles...")
    final_video = assemble_final_video(
        raw_video_path,
        updated_plan,
        srt_path,
        output_path=output_path
    )

    notify(100, "Video editing complete!")
    return final_video, updated_plan
