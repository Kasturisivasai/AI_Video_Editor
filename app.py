import os
import re
import tempfile
import pandas as pd
import streamlit as st
from pipeline import run_pipeline, write_subtitles_srt, burn_subtitles_ffmpeg, DEFAULT_GEMINI_KEY, DEFAULT_PEXELS_KEY

# Configure Streamlit Page
st.set_page_config(
    page_title="AutoReel AI — Autonomous Video Editor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Glassmorphism, Sleek Dark Mode, Modern Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main {
        background: radial-gradient(circle at 50% 0%, #171b26 0%, #0d0f15 100%);
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #A5B4FC 50%, #6366F1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 12px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        padding: 12px 28px;
        font-size: 1.05rem;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for Settings & API Keys
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    
    st.markdown("#### API Keys")
    gemini_key = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", DEFAULT_GEMINI_KEY),
        type="password",
        help="Used for transcription, translation, and edit blueprint generation."
    )
    pexels_key = st.text_input(
        "Pexels API Key",
        value=os.environ.get("PEXELS_API_KEY", DEFAULT_PEXELS_KEY),
        type="password",
        help="Used to fetch high-resolution stock photography."
    )
    
    st.markdown("---")
    st.markdown("#### Visual Card Settings")
    st.info("🎨 **Style**: Animated Pop-in + Ken Burns Drift (Podcast/Reel Edition)")
    
    st.markdown("#### Subtitle Options")
    margin_v = st.slider(
        "Subtitle Bottom Offset (px)",
        min_value=10,
        max_value=80,
        value=15,
        step=5,
        help="Adjusts vertical placement of the subtitles."
    )

# Main Hero Header
st.markdown('<div class="badge">AI-POWERED VIDEO POST-PRODUCTION</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">AutoReel Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Upload any talking-head or social video. Our autonomous AI agents transcribe, direct visual cues, fetch high-res stock imagery, and render broadcast-ready edits.</div>',
    unsafe_allow_html=True
)

# Upload Section
uploaded_file = st.file_uploader(
    "Drop your raw MP4 video here",
    type=["mp4", "mov", "m4v"],
    help="Upload vertical (9:16) or standard talking-head video."
)

if uploaded_file is not None:
    temp_dir = tempfile.mkdtemp()
    raw_video_path = os.path.join(temp_dir, "raw_input.mp4")
    output_video_path = os.path.join(temp_dir, "final_edited_video.mp4")
    
    with open(raw_video_path, "wb") as f:
        f.write(uploaded_file.read())

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📹 Raw Video")
        st.video(raw_video_path)

    with col2:
        st.markdown("### 🎬 Enhanced Output")
        
        # Check if already processed in this session
        if "processed_video" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
            st.info("Set any specific proper nouns or vocabulary hints below, then click to generate.")
            
            vocab_hints = st.text_input(
                "📝 Name / Vocabulary Hints (Optional)",
                placeholder="e.g. Dr. Hymavathi, Hyma Prasad, KIMS Hospitals",
                help="Enter comma-separated proper nouns or medical/technical terms to ensure exact spelling in subtitles."
            )
            
            start_btn = st.button("✨ Auto-Edit Video with AI")
            
            if start_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.markdown(f"**[{pct}%]** {msg}")
                
                try:
                    with st.spinner("AI Agents at work..."):
                        final_path, plan, transcript_data = run_pipeline(
                            raw_video_path=raw_video_path,
                            output_path=output_video_path,
                            gemini_key=gemini_key,
                            pexels_key=pexels_key,
                            vocab_hints=vocab_hints,
                            progress_callback=update_progress
                        )
                        st.session_state["processed_video"] = final_path
                        st.session_state["current_file"] = uploaded_file.name
                        st.session_state["edit_plan"] = plan
                        st.session_state["transcript_data"] = transcript_data
                        st.session_state["output_path"] = output_video_path
                        st.rerun()
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")
        else:
            final_path = st.session_state["processed_video"]
            if os.path.exists(final_path):
                st.video(final_path)
                with open(final_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Edited MP4",
                        data=f.read(),
                        file_name="final_edited_video.mp4",
                        mime="video/mp4"
                    )
                st.success("🎉 Video rendered with animated photo cards and desk-aligned subtitles!")

    # Subtitle Review & Live Editor
    if "transcript_data" in st.session_state and st.session_state.get("processed_video"):
        st.markdown("---")
        with st.expander("📝 Subtitle Review & Instant Editor (Fix Names & Typos)", expanded=True):
            st.markdown(
                "Notice a misspelled name or proper noun (e.g. *Hema Prasad* $\\to$ *Hyma Prasad*)? "
                "Edit below and click **Re-Burn Subtitles** to update the video in ~1-2 seconds without re-rendering the whole video!"
            )
            
            # Find & Replace Tool
            st.markdown("##### 🔍 Quick Find & Replace")
            fr_col1, fr_col2, fr_col3 = st.columns([3, 3, 2])
            with fr_col1:
                find_txt = st.text_input("Find text", placeholder="e.g. Hema Prasad", key="fr_find")
            with fr_col2:
                replace_txt = st.text_input("Replace with", placeholder="e.g. Hyma Prasad", key="fr_replace")
            with fr_col3:
                st.write("")
                st.write("")
                if st.button("Apply Replacement", key="btn_apply_fr"):
                    if find_txt:
                        count = 0
                        for seg in st.session_state["transcript_data"].get("segments", []):
                            if find_txt.lower() in seg["text"].lower():
                                seg["text"] = re.sub(re.escape(find_txt), replace_txt, seg["text"], flags=re.IGNORECASE)
                                count += 1
                        st.success(f"Replaced {count} occurrences of '{find_txt}' with '{replace_txt}'!")
                        st.rerun()
            
            # Interactive Data Editor Table
            st.markdown("##### ✏️ Interactive Subtitle Table")
            segments = st.session_state["transcript_data"].get("segments", [])
            df = pd.DataFrame([
                {"ID": s["id"], "Start": f"{s['start']:.1f}s", "End": f"{s['end']:.1f}s", "Subtitle Text": s["text"]}
                for s in segments
            ])
            edited_df = st.data_editor(
                df,
                column_config={
                    "ID": st.column_config.NumberColumn(disabled=True, width="small"),
                    "Start": st.column_config.TextColumn(disabled=True, width="small"),
                    "End": st.column_config.TextColumn(disabled=True, width="small"),
                    "Subtitle Text": st.column_config.TextColumn(width="large")
                },
                use_container_width=True,
                num_rows="fixed",
                key="sub_editor"
            )
            
            # Fast Re-Burn Button
            if st.button("⚡ Re-Burn Subtitles into Video (Takes ~2s)", key="btn_reburn"):
                # Update segments from dataframe
                for idx, row in edited_df.iterrows():
                    if idx < len(segments):
                        segments[idx]["text"] = row["Subtitle Text"]
                
                # Write updated srt
                updated_srt = "subtitles.srt"
                write_subtitles_srt(st.session_state["transcript_data"], updated_srt)
                
                # Fast re-burn using FFmpeg on temp_assembled.mp4
                base_video = "temp_assembled.mp4" if os.path.exists("temp_assembled.mp4") else raw_video_path
                target_output = st.session_state.get("output_path", "final_edited_video.mp4")
                
                burn_subtitles_ffmpeg(
                    video_input_path=base_video,
                    srt_path=updated_srt,
                    video_output_path=target_output,
                    margin_v=margin_v
                )
                st.session_state["processed_video"] = target_output
                st.success("✨ Subtitles updated and re-burned into video in ~1.5 seconds!")
                st.rerun()

    # Inspect AI Blueprint
    if "edit_plan" in st.session_state:
        st.markdown("---")
        with st.expander("🔍 Inspect AI Director's Edit Blueprint"):
            st.json(st.session_state["edit_plan"])
