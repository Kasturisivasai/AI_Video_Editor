import glob
import json
import os
import shutil
import subprocess
from faster_whisper import WhisperModel

def get_ffmpeg_path() -> str:
    """Finds ffmpeg executable on PATH or in standard Windows installation directories."""
    # 1. Check if directly available on PATH
    path = shutil.which("ffmpeg")
    if path:
        return path

    # 2. Check WinGet package directories
    winget_pattern = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\*ffmpeg*\**\ffmpeg.exe")
    matches = glob.glob(winget_pattern, recursive=True)
    if matches:
        ffmpeg_dir = os.path.dirname(matches[0])
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] += os.pathsep + ffmpeg_dir
        return matches[0]

    # 3. Check common installation directories
    for common in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        os.path.expandvars(r"%ProgramFiles%\ffmpeg\bin\ffmpeg.exe"),
    ]:
        if os.path.exists(common):
            return common

    raise FileNotFoundError(
        "FFmpeg executable not found. Please ensure FFmpeg is installed and added to PATH, or restart your terminal."
    )

def extract_audio(input_video_path: str, output_audio_path: str = "temp_audio.wav") -> str:
    """Extracts a 16kHz mono WAV audio track using FFmpeg."""
    ffmpeg_exe = get_ffmpeg_path()
    command = [
        ffmpeg_exe, "-y", "-i", input_video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_audio_path

def transcribe_audio(audio_path: str, output_json_path: str = "transcript.json") -> dict:
    """Transcribes audio using faster-whisper with millisecond-level word timestamps."""
    # Use 'small' or 'medium' for higher accuracy in Telugu; 'base' for faster testing
    model = WhisperModel("small", device="cuda", compute_type="float16")
    
    segments, info = model.transcribe(
        audio_path,
        language="te",
        word_timestamps=True,
        vad_filter=True  # Strips out pure silence blocks automatically
    )
    
    transcript_data = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": []
    }
    
    for segment in segments:
        seg_entry = {
            "id": segment.id,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
            "words": [
                {
                    "word": w.word.strip(),
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                    "probability": round(w.probability, 2)
                }
                for w in segment.words
            ] if segment.words else []
        }
        transcript_data["segments"].append(seg_entry)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)

    print(f"Transcription saved to {output_json_path}")
    return transcript_data

if __name__ == "__main__":
    # Place your sample video in the workspace and rename or point to it here:
    video_file = r"C:\Users\ksska\OneDrive\Desktop\Ollama\Edit project\VID-20260904-WA0005.mp4" 
    if not os.path.exists(video_file):
        video_file = "VID-20260904-WA0005.mp4"

    if os.path.exists(video_file):
        print(f"Extracting audio from {video_file}...")
        audio_file = extract_audio(video_file)
        print("Starting transcription with faster-whisper...")
        transcribe_audio(audio_file)
    else:
        print(f"File not found: {video_file}. Move the video into the project root.")