import json
import os
import re
import requests

from dotenv import load_dotenv

load_dotenv()

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
DOWNLOAD_DIR = "assets/b_roll"

def clean_query(query: str) -> str:
    """Cleans punctuation and limits query length for better Pexels search results."""
    words = re.sub(r"[^\w\s]", "", query).split()
    # Pexels matches best with 2 to 4 keywords
    return " ".join(words[:4]) if len(words) > 4 else " ".join(words)

def search_and_download_photo(query: str, target_filename: str):
    """Searches Pexels Photos API for high-resolution stock imagery and downloads it."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(DOWNLOAD_DIR, target_filename)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"Photo asset already exists: {dest_path}")
        return dest_path

    headers = {"Authorization": PEXELS_API_KEY}
    search_terms = [query, clean_query(query), query.split()[0] if query.split() else "abstract"]

    for term in search_terms:
        url = f"https://api.pexels.com/v1/search?query={term}&per_page=1"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                photos = data.get("photos", [])
                if photos:
                    src_dict = photos[0].get("src", {})
                    # Prefer high-res vertical/portrait or large image
                    img_url = src_dict.get("large2x") or src_dict.get("portrait") or src_dict.get("large") or src_dict.get("original")
                    if img_url:
                        print(f"Downloading stock photo for '{term}'...")
                        img_data = requests.get(img_url, timeout=15).content
                        with open(dest_path, "wb") as f:
                            f.write(img_data)
                        print(f"Saved photo to: {dest_path}")
                        return dest_path
        except Exception as e:
            print(f"Warning: Failed to fetch photo for '{term}': {e}")

    print(f"No stock photo found for '{query}'")
    return None

def search_and_download_video(query: str, target_filename: str):
    """Searches Pexels Video API as fallback if video is preferred."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(DOWNLOAD_DIR, target_filename)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"Video asset already exists: {dest_path}")
        return dest_path

    headers = {"Authorization": PEXELS_API_KEY}
    url = f"https://api.pexels.com/videos/search?query={clean_query(query)}&per_page=1"
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            videos = resp.json().get("videos", [])
            if videos:
                for f_info in videos[0].get("video_files", []):
                    if f_info.get("file_type") == "video/mp4" and f_info.get("quality") == "hd":
                        link = f_info.get("link")
                        if link:
                            print(f"Downloading stock video for '{query}'...")
                            vid_data = requests.get(link, timeout=20).content
                            with open(dest_path, "wb") as f:
                                f.write(vid_data)
                            print(f"Saved video to: {dest_path}")
                            return dest_path
    except Exception as e:
        print(f"Warning: Failed to fetch video for '{query}': {e}")

    return None

def download_b_roll_from_plan(edit_plan_path="edit_plan.json"):
    """Reads edit_plan.json and fetches stock pictures (or videos) for all visual cues."""
    if not os.path.exists(edit_plan_path):
        print(f"Error: {edit_plan_path} not found.")
        return

    with open(edit_plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    # Support both visual_cues (new universal schema) and b_roll_cues (legacy)
    cues = plan.get("visual_cues") or plan.get("b_roll_cues") or []
    print(f"Found {len(cues)} visual cue(s) in edit plan.")

    for idx, cue in enumerate(cues):
        keyword = cue.get("search_keyword", "interview")
        asset_type = cue.get("asset_type", "photo")
        slug = re.sub(r"[^\w]", "_", keyword.strip().lower())[:30]

        if asset_type == "video":
            filename = f"asset_{idx}_{slug}.mp4"
            saved_path = search_and_download_video(keyword, filename)
            if not saved_path:
                saved_path = search_and_download_photo(keyword, f"asset_{idx}_{slug}.jpg")
        else:
            # Default to high-quality photo
            filename = f"asset_{idx}_{slug}.jpg"
            saved_path = search_and_download_photo(keyword, filename)
            if not saved_path:
                saved_path = search_and_download_video(keyword, f"asset_{idx}_{slug}.mp4")

        cue["local_file"] = saved_path

    # Save back to edit_plan.json
    with open(edit_plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    print("Updated edit_plan.json with local asset paths.")

if __name__ == "__main__":
    download_b_roll_from_plan()