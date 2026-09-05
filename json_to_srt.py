import json
import os

def format_srt_time(seconds):
    """Converts float seconds to SRT time format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_srt(json_path="transcript_english.json", srt_path="subtitles.srt"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} missing.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(data.get("segments", [])):
            start = format_srt_time(segment["start"])
            end = format_srt_time(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i+1}\n{start} --> {end}\n{text}\n\n")
            
    print(f"Success: Subtitles saved to {srt_path}")

if __name__ == "__main__":
    generate_srt()