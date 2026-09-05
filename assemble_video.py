import json
import os
import PIL.Image
import numpy as np

# Compatibility patch for MoviePy 1.0.3 with Pillow >= 10.0
if not hasattr(PIL.Image, "ANTIALIAS"):
    setattr(PIL.Image, "ANTIALIAS", PIL.Image.Resampling.LANCZOS)

from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from moviepy.video.fx.crop import crop

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

def assemble_cut(
    raw_video_path="VID-20260904-WA0005.mp4",
    plan_path="edit_plan.json",
    output_path="final_edited_video.mp4"
):
    if not os.path.exists(raw_video_path) or not os.path.exists(plan_path):
        print("Missing raw video or edit_plan.json!")
        return

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    print("Loading raw video...")
    main_clip = VideoFileClip(raw_video_path)
    if main_clip.duration is None:
        print("Could not determine video duration!")
        return

    # Keep the full continuous video without dropping any footage
    combined = main_clip

    # Layer Visual Asset Overlays (Photos & B-Roll)
    visual_cues = plan.get("visual_cues") or plan.get("b_roll_cues") or []
    overlays = [combined]

    for cue in visual_cues:
        local_path = cue.get("local_file")
        start = cue.get("start", 0)
        end = cue.get("end", 0)
        duration = end - start

        effective_start = float(start)
        if local_path and os.path.exists(local_path) and duration > 0:
            is_image = local_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            print(f"Overlaying Animated Photo Card '{cue.get('search_keyword')}' at {effective_start:.2f}s - {(effective_start + duration):.2f}s...")
            
            if is_image:
                card_clip = make_animated_card_clip(local_path, duration, combined.w, combined.h)
                card_clip = card_clip.set_start(effective_start)
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
                asset_clip = asset_clip.set_position("center").set_start(effective_start)
                overlays.append(asset_clip)

    final_render = CompositeVideoClip(overlays)

    print("Rendering base video without subtitles...")
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
    
    # Release clip file handles
    final_render.close()
    main_clip.close()
    
    print("Burning subtitles via FFmpeg with sleek modern typography...")
    import subprocess
    import shutil
    import imageio_ffmpeg

    ffmpeg_exe = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    
    # Ensure subtitles.srt exists
    srt_path = "subtitles.srt"
    if not os.path.exists(srt_path):
        import json_to_srt
        json_to_srt.generate_srt()

    # Modern Instagram / Podcast subtitle style:
    # Bold Arial font, crisp white text, semi-transparent background box, resting below desk title
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
        "MarginV=15"
    )
    
    # Replace backslashes with forward slashes for FFmpeg path compatibility
    safe_srt_path = srt_path.replace("\\", "/")
    ffmpeg_cmd = [
        ffmpeg_exe, "-y", "-i", temp_output, 
        "-vf", f"subtitles='{safe_srt_path}':force_style='{style_opts}'", 
        "-c:a", "copy", output_path
    ]
    
    subprocess.run(ffmpeg_cmd, check=True)
    
    # Clean up temporary files safely
    for temp_f in [temp_output, "temp_assembledTEMP_MPY_wvf_snd.mp4", "test_card.png"]:
        if os.path.exists(temp_f):
            try:
                os.remove(temp_f)
            except Exception:
                pass
        
    print(f"Done! Final edited video saved as {output_path}")

if __name__ == "__main__":
    assemble_cut()