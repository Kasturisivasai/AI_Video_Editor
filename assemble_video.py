import json
import os
import PIL.Image

# Compatibility patch for MoviePy 1.0.3 with Pillow >= 10.0
if not hasattr(PIL.Image, "ANTIALIAS"):
    setattr(PIL.Image, "ANTIALIAS", PIL.Image.Resampling.LANCZOS)

from moviepy.editor import VideoFileClip
from pipeline import assemble_final_video, write_subtitles_ass, extract_curated_punch_callouts

def assemble_cut(
    raw_video_path="VID-20260904-WA0005.mp4",
    plan_path="edit_plan.json",
    transcript_path="transcript_english.json",
    output_path="final_edited_video.mp4"
):
    if not os.path.exists(raw_video_path) or not os.path.exists(plan_path):
        print("Missing raw video or edit_plan.json!")
        return

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    transcript_data = None
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

    print("Probing video dimensions...")
    probe_clip = VideoFileClip(raw_video_path)
    vid_w, vid_h = probe_clip.w, probe_clip.h
    probe_clip.close()

    highlight_words = "toxic, double, depression, money, salary, manager"
    punch_callouts = extract_curated_punch_callouts(plan, transcript_data, highlight_words)

    print("Generating reel ASS subtitles with in-line gold word highlights & center kinetic callouts...")
    ass_path = write_subtitles_ass(
        transcript_data,
        ass_path="subtitles.ass",
        video_w=vid_w,
        video_h=vid_h,
        highlight_words=highlight_words,
        margin_v=55,
        punch_callouts=punch_callouts
    )

    print("Assembling video with upper photo cards, kinetic reel typography & subtitles...")
    assemble_final_video(
        raw_video_path=raw_video_path,
        edit_plan=plan,
        sub_path=ass_path,
        output_path=output_path,
        transcript_data=transcript_data,
        highlight_words=highlight_words,
        sub_margin_v=55,
        punch_callouts=punch_callouts
    )
    print(f"Done! Final edited video saved as {output_path}")

if __name__ == "__main__":
    assemble_cut()