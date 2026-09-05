[Client: Mother's Web Browser]
     | (1) Upload Raw MP4
     v
[Frontend Portal: Streamlit] ---> (Temporary File Storage)
     | (2) Trigger Pipeline
     v
[Audio Processing Module]
     | (3) Extract Audio Track (FFmpeg)
     v
[Transcription Engine: Groq API / Whisper]
     | (4) Return Timestamped JSON Transcript (Telugu & English)
     v
[Editorial Brain: Gemini API]
     | (5) Input: Transcript + System Prompt
     | (6) Output: JSON schema {Hook_Time, Cuts, B-Roll_Cues, Captions}
     v
[Asset Retrieval Script]
     | (7) Query Free APIs (Pexels) using B-Roll Cues
     | (8) Download relevant Stock MP4s
     v
[Video Assembly Engine: Python + FFmpeg]
     | (9) Ingest Raw MP4, Stock MP4s, and JSON Edit Data
     | (10) Execute jump cuts, zoom overlays, and text burn-in
     v
[Final Output]
     | (11) Render final Edited MP4
     v
[Frontend Portal] ---> (Download to Client)