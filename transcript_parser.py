import pandas as pd
import os
import re
import json
from typing import List, Dict
from datetime import datetime, time

def _parse_structured_json(data: List[Dict]) -> List[Dict]:
    """Handles the specific, high-quality JSON format with UNIX timestamps."""
    if not data:
        return []

    try:
        call_start_time = min(item.get('start_timestamp', float('inf')) for item in data)
    except (TypeError, ValueError):
        return []

    segments = []
    for item in data:
        start_ts = item.get('start_timestamp', 0)
        end_ts = item.get('end_timestamp', 0)
        
        start_relative = start_ts - call_start_time
        end_relative = end_ts - call_start_time
        
        segments.append({
            "start": float(start_relative),
            "end": float(end_relative),
            "text": item.get("text", "").strip(),
            "speaker": item.get("name", "Unknown Speaker")
        })
    return segments

def _parse_structured_csv(file_path: str) -> List[Dict]:
    """
    Handles a structured CSV file with human-readable timestamps.
    """
    df = pd.read_csv(file_path)
    
    required_cols = {'start_timestamp', 'end_timestamp', 'text', 'name'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV file must contain the columns: {', '.join(required_cols)}")

    # Convert time strings to datetime objects 
    start_times = pd.to_datetime(df['start_timestamp'], format='%I:%M:%S %p').dt.time
    call_start_time = min(start_times)
    
    # Convert the start time to a datetime object
    start_datetime = datetime.combine(datetime.today(), call_start_time)

    segments = []
    for _, row in df.iterrows():
        # Calculate start and end seconds relative to the call start time
        start_dt_row = datetime.combine(datetime.today(), pd.to_datetime(row['start_timestamp'], format='%I:%M:%S %p').time())
        end_dt_row = datetime.combine(datetime.today(), pd.to_datetime(row['end_timestamp'], format='%I:%M:%S %p').time())

        start_relative = (start_dt_row - start_datetime).total_seconds()
        end_relative = (end_dt_row - start_datetime).total_seconds()

        segments.append({
            "start": float(start_relative),
            "end": float(end_relative),
            "text": row.get("text", "").strip(),
            "speaker": row.get("name", "Unknown Speaker")
        })
    return segments

def _parse_semi_structured_txt(file_path: str) -> List[Dict]:
    """
    Handles a TXT file with a specific 'Speaker | Timestamp -->' format using regex.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to capture speaker, timestamp, and the text that follows
    pattern = re.compile(r"^(Speaker\d+)\s*\|\s*([\d: ]+\s*[AP]M)\s*-->\n(.*?)(?=\n\nSpeaker\d+|$)", re.MULTILINE | re.DOTALL)
    matches = pattern.findall(content)

    if not matches:
        raise ValueError("Could not find any valid speaker segments in the TXT file. The format should be 'SpeakerX | HH:MM:SS AM/PM -->'.")

    # Process matches to calculate relative timestamps
    parsed_segments = []
    start_times = [datetime.strptime(m[1].strip(), '%I:%M:%S %p').time() for m in matches]
    call_start_time = min(start_times)
    start_datetime = datetime.combine(datetime.today(), call_start_time)

    for i, match in enumerate(matches):
        speaker, time_str, text = match
        
        # Calculate start time in seconds
        current_dt = datetime.combine(datetime.today(), datetime.strptime(time_str.strip(), '%I:%M:%S %p').time())
        start_seconds = (current_dt - start_datetime).total_seconds()
        
        # Estimate end time based on the start of the next segment
        if i + 1 < len(matches):
            next_time_str = matches[i+1][1]
            next_dt = datetime.combine(datetime.today(), datetime.strptime(next_time_str.strip(), '%I:%M:%S %p').time())
            end_seconds = (next_dt - start_datetime).total_seconds()
        else:
            # For the last segment, estimate duration based on text length (e.g., 5 seconds per 20 words)
            word_count = len(text.strip().split())
            estimated_duration = (word_count / 20) * 5
            end_seconds = start_seconds + max(5, estimated_duration) # Minimum 5 seconds

        parsed_segments.append({
            "start": float(start_seconds),
            "end": float(end_seconds),
            "text": text.strip(),
            "speaker": speaker
        })
    return parsed_segments

def parse_transcript_file(file_path: str) -> List[Dict]:
    """
    Reads a transcript file and intelligently converts it into the standard segment format.
    """
    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension == '.json':
        print("Parsing structured JSON file.")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return _parse_structured_json(data)

    elif file_extension == '.csv':
        print("Parsing structured CSV file.")
        return _parse_structured_csv(file_path)

    elif file_extension == '.txt':
        print("Parsing semi-structured TXT file with regex.")
        return _parse_semi_structured_txt(file_path)
            
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
