import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import the microphone recorder component
from streamlit_mic_recorder import mic_recorder

# Import your custom modules
from normalization import transcribe_audio
from ingest import ingest_segments_to_chroma
from query_engine import choose_tool, query_chroma
from profiler import create_conversation_profile
from outputs import (
    build_text_response,
    build_llm_chart_response,
    build_speaker_activity_chart,
    build_speaker_turn_count_chart,
    build_sentiment_trend_chart,
    build_holistic_analysis_chart,
    build_audio_response
)
from transcript_parser import parse_transcript_file
from pdf_builder import build_pdf_response

# --- App Configuration ---
st.set_page_config(page_title="Multimodal AI Agent", layout="wide")
st.title("Multimodal AI Agent for Enterprise Conversations 🚀")

# --- Define Available Tools for the Router ---
tools = [
    {"name": "create_holistic_analysis_chart", "description": "Performs a complex, deep analysis of the entire transcript. Use for broad, analytical questions about topics, time allocation, categorization, or summarization that require understanding the full context of the conversation."},
    {"name": "create_speaker_turn_count_chart", "description": "Generates a bar chart counting the NUMBER OF TIMES each speaker spoke. Use for questions like 'How many times did each speaker talk?' or 'Count speaker turns'."},
    {"name": "create_speaker_activity_chart", "description": "Generates a bar chart showing the total DURATION (in seconds) of each speaker's talk time. Use for questions about who spoke the most, talk time distribution, or speaker activity."},
    {"name": "create_sentiment_trend_chart", "description": "Creates a line chart showing the sentiment of a speaker or the conversation over time. Use for questions about mood, sentiment, or tone."},
    {"name": "create_keyword_mention_chart", "description": "Generates a chart counting the frequency of specific, simple keywords. Use for questions like 'How often was X mentioned?'"},
    {"name": "summarize_text", "description": "Creates a text summary of the conversation. This is the default tool for general questions."},
    {"name": "generate_pdf_report", "description": "Creates a downloadable PDF document summarizing the key findings related to a query. Use for requests to 'download a report'."},
    {"name": "generate_audio_summary", "description": "Generates an audio narration of a summary. Use for requests to 'read the summary aloud'."}
]

# --- Persistent State Management ---
PROFILE_FILE = "active_profile.json"
TRANSCRIPT_CACHE_FILE = "transcript_cache.json"

if "current_profile" not in st.session_state:
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f: st.session_state.current_profile = json.load(f)
        if os.path.exists(TRANSCRIPT_CACHE_FILE):
            with open(TRANSCRIPT_CACHE_FILE, "r") as f: st.session_state.full_transcript_segments = json.load(f)
    else:
        st.session_state.current_profile = None
        st.session_state.full_transcript_segments = []

# --- Sidebar for Data Ingestion ---
st.sidebar.header("Process Call Data")
st.sidebar.write("Upload a pre-existing file or record a new conversation.")

# Option 1: File Uploader
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["mp3", "wav", "txt", "csv", "json"])
if uploaded_file is not None:
    file_path = os.path.join("temp_uploads", uploaded_file.name)
    os.makedirs("temp_uploads", exist_ok=True)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    if st.sidebar.button("Process Uploaded File"):
        with st.spinner("Processing uploaded file..."):
            try:
                # (Full processing pipeline logic)
                file_extension = uploaded_file.name.split('.')[-1].lower()
                st.sidebar.info(f"Step 1/3: Parsing/Transcribing {file_extension}...")
                segments = parse_transcript_file(file_path) if file_extension in ["txt", "csv", "json"] else transcribe_audio(file_path)
                st.sidebar.info("Step 2/3: Ingesting data...")
                ingest_segments_to_chroma(segments, file_id=uploaded_file.name)
                st.sidebar.info("Step 3/3: Profiling conversation...")
                full_text = " ".join([seg['text'] for seg in segments])
                profile = create_conversation_profile(full_text)
                profile['source_file'] = uploaded_file.name
                st.session_state.current_profile = profile
                st.session_state.full_transcript_segments = segments
                with open(PROFILE_FILE, "w") as f: json.dump(profile, f)
                with open(TRANSCRIPT_CACHE_FILE, "w") as f: json.dump(segments, f)
                st.sidebar.success("File processed and profiled!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}")

st.sidebar.divider()

# Option 2: Live Conversation Recording
st.sidebar.write("Record a new conversation:")
convo_audio = mic_recorder(start_prompt="🔴 Record Conversation", stop_prompt="⏹️ Stop", key='convo_recorder')

if convo_audio:
    st.sidebar.audio(convo_audio['bytes'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join("temp_uploads", f"recording_{timestamp}.wav")
    os.makedirs("temp_uploads", exist_ok=True)
    with open(file_path, "wb") as f: f.write(convo_audio['bytes'])
    if st.sidebar.button("Process Recording"):
        with st.spinner("Processing recording..."):
            try:
                # (Full processing pipeline logic for the recording)
                st.sidebar.info("Step 1/3: Transcribing audio...")
                segments = transcribe_audio(file_path)
                st.sidebar.info("Step 2/3: Ingesting data...")
                ingest_segments_to_chroma(segments, file_id=f"recording_{timestamp}.wav")
                st.sidebar.info("Step 3/3: Profiling conversation...")
                full_text = " ".join([seg['text'] for seg in segments])
                profile = create_conversation_profile(full_text)
                profile['source_file'] = f"recording_{timestamp}.wav"
                st.session_state.current_profile = profile
                st.session_state.full_transcript_segments = segments
                with open(PROFILE_FILE, "w") as f: json.dump(profile, f)
                with open(TRANSCRIPT_CACHE_FILE, "w") as f: json.dump(segments, f)
                st.sidebar.success("Recording processed and profiled!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}")

st.sidebar.divider()
if st.session_state.current_profile:
    st.sidebar.write("Current Active Profile:")
    st.sidebar.json(st.session_state.current_profile)

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def display_chart(response):
    # (This function is unchanged)
    if "error" not in response:
        data = response.get("data")
        is_empty = False
        if data is None: is_empty = True
        elif isinstance(data, pd.Series): is_empty = data.empty
        elif isinstance(data, dict): is_empty = not data
        if is_empty:
            st.warning("The AI could not find any data to plot for this query.")
            return
        chart_type = response.get("chart_type")
        fig, ax = plt.subplots()
        if chart_type == "bar":
            df = pd.DataFrame.from_dict(data, orient='index', columns=['value'])
            df.plot(kind='bar', ax=ax, legend=False); plt.xticks(rotation=45, ha="right")
        elif chart_type == "pie":
            ax.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', startangle=90); ax.axis('equal')
        elif chart_type == "line":
            data.plot(kind='line', ax=ax, legend=False)
            plt.axhline(0, color='grey', linewidth=0.8)
        ax.set_title(response.get("title", "Chart"))
        ax.set_xlabel(response.get("x_label", ""))
        ax.set_ylabel(response.get("y_label", ""))
        plt.tight_layout(); st.pyplot(fig)
    else:
        st.error(response.get("error"))

# --- NEW: Voice Query Logic ---
# We use columns to place the voice input next to the text input
col1, col2 = st.columns([.9, .1])
with col1:
    prompt = st.chat_input("Ask anything about your calls...")
with col2:
    st.write(" ") # Spacer
    query_audio = mic_recorder(
        start_prompt="🎤", 
        stop_prompt="⏹️", 
        key='query_recorder',
        use_container_width=True
    )

if query_audio:
    # Transcribe the short voice query
    with st.spinner("Transcribing your question..."):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join("temp_uploads", f"query_{timestamp}.wav")
        with open(file_path, "wb") as f:
            f.write(query_audio['bytes'])
        # Use a smaller, faster model for short queries
        segments = transcribe_audio(file_path, model_size="tiny.en") 
        prompt = " ".join([s['text'] for s in segments])
        os.remove(file_path) # Clean up the temporary query file

# --- Main Application Logic ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            profile = st.session_state.get("current_profile")
            if not profile:
                st.warning("Please process a file first."); st.stop()

            chosen_tool = choose_tool(prompt, tools, profile)
            
            if chosen_tool == "create_holistic_analysis_chart":
                full_transcript = st.session_state.get("full_transcript_segments", [])
                response = build_holistic_analysis_chart(prompt, full_transcript)
                display_chart(response)
            else:
                top_k = 75 if any(k in chosen_tool for k in ["activity", "sentiment", "turn_count"]) else 20
                retrieved = query_chroma(prompt, top_k=top_k)

                if "chart" in chosen_tool:
                    if chosen_tool == "create_speaker_turn_count_chart":
                        response = build_speaker_turn_count_chart(retrieved)
                    elif chosen_tool == "create_sentiment_trend_chart":
                        response = build_sentiment_trend_chart(prompt, retrieved)
                    elif chosen_tool == "create_speaker_activity_chart":
                        response = build_speaker_activity_chart(retrieved)
                    else:
                        response = build_llm_chart_response(prompt, retrieved)
                    display_chart(response)
                
                elif chosen_tool == "generate_pdf_report":
                    response = build_pdf_response(prompt, retrieved, profile)
                    with open(response["pdf_path"], "rb") as f:
                        st.download_button("Download Report", f, "report.pdf", "application/pdf")
                elif chosen_tool == "generate_audio_summary":
                    text_response = build_text_response(prompt, retrieved)
                    summary_text = text_response.get("text_summary")
                    if summary_text and not summary_text.startswith("Error:"):
                        audio_response = build_audio_response(summary_text)
                        st.audio(audio_response["audio_path"])
                    else: st.error("Could not generate audio.")
                else: # Catches "summarize_text"
                    response = build_text_response(prompt, retrieved)
                    st.markdown(response.get("text_summary", "Sorry, I couldn't find an answer."))

    st.session_state.messages.append({"role": "assistant", "content": "Done."})