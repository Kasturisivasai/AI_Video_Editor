# 🎬 AutoReel AI Studio

> **Transform raw talking-head and podcast videos into engaging, viral short-form social reels automatically using autonomous AI agents.**

AutoReel AI Studio is an end-to-end video post-production engine. It extracts speech, translates it to crisp English captions, generates narrative visual cues, retrieves high-resolution contextual stock photography, and composites professional animated photo cards and collision-free subtitles—all without manual timeline editing.

---

## ✨ Features

- 🎙️ **Multilingual Audio Transcription & Translation**: Powered by Google Gemini Flash to accurately transcribe any spoken language and produce punchy, conversational English subtitles.
- 🧠 **Autonomous Editorial Director Agent**: Analyzes narrative flow and timestamps to identify high-retention visual moments, metaphors, and conceptual cues.
- 🖼️ **Dynamic Pexels Asset Retrieval**: Queries and downloads high-res contextual stock photography tailored to the speaker's exact talking points.
- 🎨 **Broadcast-Quality Photo Card Animations**:
  - **Pop-In Zoom Entry**: Smooth ease-out scale expansion ($0.75\times \to 1.0\times$) with alpha fade.
  - **Ken Burns Motion**: Subtle drift/zoom ($1.0\times \to 1.04\times$) while displayed to maintain dynamic momentum.
  - **Safe-Zone Framing**: Intelligently positioned in the upper third above the speaker, keeping the speaker's face, eyes, and nameplate completely unobstructed.
  - **Sleek Aesthetic**: Rounded corners, crisp white border, and soft drop shadow.
- 💬 **Collision-Free Modern Subtitles**: Single-line captions burned cleanly below the subject using FFmpeg libass styling.
- 🌐 **Interactive Streamlit Web Dashboard**: Drag-and-drop video upload, live multi-stage progress tracking, side-by-side comparison player, and direct MP4 export.

---

## 🛠️ Architecture

```
                                  Raw Video (MP4)
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Stage 1: Audio Extraction (FFmpeg)    │
                     │          16kHz Mono WAV               │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Stage 2: Gemini Flash Speech AI       │
                     │          Transcription & Translation  │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Stage 3: AI Editorial Director Agent  │
                     │          Visual Cues Blueprint        │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Stage 4: Pexels API Asset Ingestion   │
                     │          High-Res Contextual Photos   │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Stage 5: MoviePy & FFmpeg Compositor  │
                     │          • Pop-in Card Animation      │
                     │          • Ken Burns Drift Motion     │
                     │          • Safe-Zone Upper Framing    │
                     │          • Synced Subtitle Burning    │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                             final_edited_video.mp4
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) installed and available on your system `PATH`.

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AutoReel-AI-Studio.git
cd AutoReel-AI-Studio
```

### 3. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the `.env.example` file to `.env` and fill in your API keys:
```bash
cp .env.example .env
```
- **Gemini API Key**: Obtain from [Google AI Studio](https://aistudio.google.com/).
- **Pexels API Key**: Obtain a free key from [Pexels API](https://www.pexels.com/api/).

---

## 💻 Usage

### Run the Web Dashboard
Launch the interactive Streamlit studio:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser, drag and drop your talking-head MP4 video, and click **Auto-Edit Video with AI**.

### Run via Command Line / Script
```bash
python pipeline.py
```

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Select your repository, set the main file path to `app.py`.
4. In **Advanced Settings > Secrets**, configure:
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   PEXELS_API_KEY = "your-pexels-api-key"
   ```
5. Click **Deploy!** (`packages.txt` will automatically install FFmpeg on Streamlit Cloud).

---

## 📄 License
MIT License. Feel free to modify and build upon this project.
