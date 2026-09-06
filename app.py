import json
import os
import re
import shutil
import tempfile
import pandas as pd
import streamlit as st
from pipeline import (
    run_pipeline,
    write_subtitles_ass,
    write_subtitles_srt,
    burn_subtitles_ffmpeg,
    extract_curated_punch_callouts,
    DEFAULT_GEMINI_KEY,
    DEFAULT_PEXELS_KEY
)

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
    st.markdown("#### 🎨 Typography & Reel Styling")
    font_choice = st.selectbox(
        "Headline & Subtitle Font",
        options=["Montserrat Black", "League Spartan", "Arial Black"],
        index=0,
        help="Ultra-bold heavy sans-serif typeface in all-caps white text with thick black outline."
    )
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); padding: 10px 14px; border-radius: 10px; font-size: 0.82rem; border-left: 3px solid #6366F1; margin-bottom: 12px;">
        <b>Font Choice:</b> Ultra-bold heavy sans-serif<br>
        <b>Text Fill:</b> All-Caps Pure White (&H00FFFFFF)<br>
        <b>Outline / Stroke:</b> Heavy Black (Outline=3.8-4.8)<br>
        <b>Background Box:</b> Zero Container Box
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Subtitle & Visual Positioning Options")
    sub_font_size_pt = st.slider(
        "Subtitle Font Size (pt)",
        min_value=16,
        max_value=32,
        value=22,
        step=1,
        help="Default 22pt (+1 point increased for maximum clarity)."
    )
    sub_margin_v = st.slider(
        "Subtitle Vertical Position (px from bottom)",
        min_value=15,
        max_value=250,
        value=55,
        step=5,
        help="55px places subtitles cleanly across the lower desk line below the nameplate."
    )
    card_y_pct = st.slider(
        "Visual Picture Position (% from top)",
        min_value=8,
        max_value=25,
        value=14,
        step=1,
        help="14% moves pictures up by 3 points (clearing the subject's face and hair completely)."
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

raw_video_path = None
output_video_path = None
active_file_name = None

if uploaded_file is not None:
    temp_dir = tempfile.mkdtemp()
    raw_video_path = os.path.join(temp_dir, "raw_input.mp4")
    output_video_path = os.path.join(temp_dir, "final_edited_video.mp4")
    with open(raw_video_path, "wb") as f:
        f.write(uploaded_file.read())
    active_file_name = uploaded_file.name
elif os.path.exists("VID-20260904-WA0005.mp4"):
    st.info("💡 **Active Workspace Video Loaded**: `VID-20260904-WA0005.mp4`")
    raw_video_path = "VID-20260904-WA0005.mp4"
    output_video_path = "final_edited_video.mp4"
    active_file_name = "VID-20260904-WA0005.mp4"
    if "processed_video" not in st.session_state and os.path.exists("final_edited_video.mp4"):
        st.session_state["processed_video"] = "final_edited_video.mp4"
        st.session_state["current_file"] = active_file_name
        st.session_state["output_path"] = "final_edited_video.mp4"
        if os.path.exists("transcript_english.json"):
            with open("transcript_english.json", "r", encoding="utf-8") as f:
                st.session_state["transcript_data"] = json.load(f)
        if os.path.exists("edit_plan.json"):
            with open("edit_plan.json", "r", encoding="utf-8") as f:
                st.session_state["edit_plan"] = json.load(f)

if raw_video_path is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📹 Raw Video")
        st.video(raw_video_path)

    with col2:
        st.markdown("### 🎬 Enhanced Output")
        
        # Check if already processed in this session
        if "processed_video" not in st.session_state or st.session_state.get("current_file") != active_file_name:
            st.info("Set any specific proper nouns or highlight pop-out words below, then click to generate.")
            
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                vocab_hints = st.text_input(
                    "📝 Name / Vocabulary Hints (Optional)",
                    placeholder="e.g. Dr. Hymavathi, Hyma Prasad, KIMS Hospitals",
                    help="Enter comma-separated proper nouns to ensure exact spelling in subtitles."
                )
            with h_col2:
                highlight_words = st.text_input(
                    "🔥 Highlight Pop-Out Words (Optional)",
                    placeholder="e.g. toxic, double, depression, money, salary",
                    help="Words that will pop up as dynamic animated graphics above the subtitles when spoken."
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
                        final_path, plan, transcript_data, punch_callouts = run_pipeline(
                            raw_video_path=raw_video_path,
                            output_path=output_video_path,
                            gemini_key=gemini_key,
                            pexels_key=pexels_key,
                            vocab_hints=vocab_hints,
                            highlight_words=highlight_words,
                            sub_margin_v=sub_margin_v,
                            sub_font_size=sub_font_size_pt,
                            card_y_pct=card_y_pct / 100.0,
                            font_name=font_choice,
                            progress_callback=update_progress
                        )
                        st.session_state["processed_video"] = final_path
                        st.session_state["current_file"] = active_file_name
                        st.session_state["edit_plan"] = plan
                        st.session_state["transcript_data"] = transcript_data
                        st.session_state["punch_callouts"] = punch_callouts
                        st.session_state["output_path"] = output_video_path
                        st.session_state["highlight_words"] = highlight_words
                        st.session_state["font_choice"] = font_choice
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

    # Video Typography & Callout Review & Live Editors
    if "transcript_data" in st.session_state and st.session_state.get("processed_video"):
        st.markdown("---")
        
        # Ensure punch_callouts exists in session state
        if "punch_callouts" not in st.session_state:
            st.session_state["punch_callouts"] = extract_curated_punch_callouts(
                st.session_state.get("edit_plan", {}),
                st.session_state.get("transcript_data", {}),
                st.session_state.get("highlight_words", "")
            )

        # 1. Dedicated Center Kinetic Callouts Editor
        with st.expander("🎯 Center Kinetic Callouts (Middle Pop-Out Words)", expanded=True):
            st.markdown(
                "Customize the high-impact punch words that pop up in the middle of the screen (e.g. *HYMA PRASAD*, *POOR COMMUNICATION*, *FEAR OF ENGLISH*). "
                "Notice a misspelled name like **HEMA PRASAD**? Change it to **HYMA PRASAD** directly in the table, or uncheck **Show** to remove it completely!"
            )
            
            callouts = st.session_state.get("punch_callouts", [])
            callout_df_data = []
            for idx, c in enumerate(callouts):
                callout_df_data.append({
                    "Index": idx,
                    "Start (s)": float(c.get("start", 0)),
                    "End (s)": float(c.get("end", 0)),
                    "Callout Text": str(c.get("text", "")),
                    "Color Theme": str(c.get("color", "white")),
                    "Show": bool(c.get("enabled", True))
                })
                
            c_df = pd.DataFrame(callout_df_data)
            edited_callouts_df = st.data_editor(
                c_df,
                column_config={
                    "Index": st.column_config.NumberColumn(disabled=True, width="small"),
                    "Start (s)": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.2f", width="small"),
                    "End (s)": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.2f", width="small"),
                    "Callout Text": st.column_config.TextColumn(width="large", help="Word or phrase to appear in the center"),
                    "Color Theme": st.column_config.SelectboxColumn(
                        options=["white", "gold", "red", "red_gold"],
                        width="medium",
                        help="white = Pure White All-Caps Outlined (Reel Standard)"
                    ),
                    "Show": st.column_config.CheckboxColumn(width="small", help="Uncheck to hide this callout from video")
                },
                use_container_width=True,
                num_rows="dynamic",
                key="callout_editor"
            )

            # Quick Add Callout Form
            with st.form("add_callout_form", clear_on_submit=True):
                st.markdown("##### ➕ Add Custom Center Callout")
                ac_col1, ac_col2, ac_col3, ac_col4 = st.columns([2, 2, 4, 2])
                with ac_col1:
                    new_start = st.number_input("Start (s)", min_value=0.0, value=0.0, step=0.5)
                with ac_col2:
                    new_end = st.number_input("End (s)", min_value=0.0, value=2.0, step=0.5)
                with ac_col3:
                    new_text = st.text_input("Callout Text", placeholder="e.g. HYMA PRASAD")
                with ac_col4:
                    new_color = st.selectbox("Color", ["white", "gold", "red", "red_gold"])
                
                submitted_new_callout = st.form_submit_button("Add Word to Callouts")
                if submitted_new_callout and new_text.strip():
                    callouts.append({
                        "start": round(new_start, 2),
                        "end": round(new_end, 2),
                        "text": new_text.strip().upper(),
                        "color": new_color,
                        "enabled": True
                    })
                    st.session_state["punch_callouts"] = callouts
                    st.success(f"Added '{new_text.strip().upper()}' to callouts!")
                    st.rerun()

        # 2. Dialogue Subtitles Editor
        with st.expander("📝 Dialogue Subtitles (Lower Desk & Transcription)", expanded=True):
            st.markdown(
                "Review full speech transcription. Edit lines below or use Quick Find & Replace to fix proper nouns."
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
                        # Also replace in punch_callouts if present
                        for c in st.session_state.get("punch_callouts", []):
                            if find_txt.lower() in c.get("text", "").lower():
                                c["text"] = re.sub(re.escape(find_txt), replace_txt, c["text"], flags=re.IGNORECASE).upper()
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

        # 3. Master Re-Burn Button for Instant 1.5s Rendering
        st.markdown("---")
        if st.button("⚡ Re-Burn Video (Subtitles & Center Callouts) (Takes ~1.5s)", key="btn_reburn", type="primary"):
            # Update segments from dataframe
            for idx, row in edited_df.iterrows():
                if idx < len(segments):
                    segments[idx]["text"] = row["Subtitle Text"]

            # Update punch callouts from dataframe
            updated_callouts = []
            for _, row in edited_callouts_df.iterrows():
                txt = str(row["Callout Text"]).strip()
                if txt:
                    updated_callouts.append({
                        "start": float(row["Start (s)"]),
                        "end": float(row["End (s)"]),
                        "text": txt.upper(),
                        "color": str(row["Color Theme"]),
                        "enabled": bool(row["Show"])
                    })
            st.session_state["punch_callouts"] = updated_callouts

            # Write updated ASS with dynamic styling for both layers
            updated_ass = "subtitles.ass"
            hl_words = st.session_state.get("highlight_words", "")
            chosen_font = st.session_state.get("font_choice", font_choice)
            write_subtitles_ass(
                st.session_state["transcript_data"],
                ass_path=updated_ass,
                font_name=chosen_font,
                sub_font_size=sub_font_size_pt,
                highlight_words=hl_words,
                margin_v=sub_margin_v,
                punch_callouts=updated_callouts
            )
            write_subtitles_srt(st.session_state["transcript_data"], "subtitles.srt")
            
            # Fast re-burn using FFmpeg on temp_assembled.mp4
            base_video = "temp_assembled.mp4" if os.path.exists("temp_assembled.mp4") else raw_video_path
            target_output = st.session_state.get("output_path", "final_edited_video.mp4")
            
            burn_subtitles_ffmpeg(
                video_input_path=base_video,
                sub_path=updated_ass,
                video_output_path=target_output,
                margin_v=sub_margin_v
            )
            if target_output != "final_edited_video.mp4":
                try:
                    shutil.copy(target_output, "final_edited_video.mp4")
                except Exception:
                    pass
            st.session_state["processed_video"] = target_output
            st.success("✨ Video updated and re-burned with your exact words in ~1.5 seconds!")
            st.rerun()

    # Inspect AI Blueprint
    if "edit_plan" in st.session_state:
        st.markdown("---")
        with st.expander("🔍 Inspect AI Director's Edit Blueprint"):
            st.json(st.session_state["edit_plan"])
