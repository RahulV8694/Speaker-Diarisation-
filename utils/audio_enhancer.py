from pydub import AudioSegment, effects
import numpy as np
import os

def enhance_audio(audio_path):
    print("Loading audio file...")
    audio = AudioSegment.from_file(audio_path)
    
    print("Applying basic audio enhancements...")
    
    # Simple normalization
    normalized_audio = effects.normalize(audio)
    
    # Basic filtering for speech frequencies
    filtered_audio = normalized_audio.high_pass_filter(80)
    filtered_audio = filtered_audio.low_pass_filter(8000)
    
    # Gentle compression
    filtered_audio = effects.compress_dynamic_range(filtered_audio, threshold=-20, ratio=3)
    
    # Final normalization
    filtered_audio = effects.normalize(filtered_audio)
    
    enhanced_path = os.path.splitext(audio_path)[0] + "_enhanced.wav"
    print(f"Saving enhanced audio to: {enhanced_path}")
    filtered_audio.export(enhanced_path, format="wav", parameters=["-ar", "16000"])
    
    return enhanced_path