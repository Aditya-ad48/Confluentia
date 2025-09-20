from faster_whisper import WhisperModel
from typing import List, Dict

_whisper_cache = {}

def transcribe_audio(audio_path: str, model_size: str = "base") -> List[Dict]:
    """
    Transcribes audio from any supported language and translates it into English.
    This creates a single, consistent language for the rest of the AI pipeline.
    """
    print(f"Loading Whisper model '{model_size}' for transcription and translation...")
    
    # Load model 
    if model_size not in _whisper_cache:
        _whisper_cache[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    model = _whisper_cache[model_size]

  
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        task="translate", 
        word_timestamps=True
    )
    
    print(f"Detected source language: {info.language} (Confidence: {info.language_probability:.2f})")
    
    results = []
   
    for segment in segments:
    
        # Assign speakers in a round fashion (SPEAKER_0, SPEAKER_1, etc.)
        speaker = f"SPEAKER_{len(results) % 2}" 
        
        # Reconstruct the text from the word-level data
        segment_text = "".join(word.word for word in segment.words).strip()
        
        results.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment_text,
            "speaker": speaker
        })
        
    print(f"Successfully transcribed and translated audio into {len(results)} segments.")
    return results
