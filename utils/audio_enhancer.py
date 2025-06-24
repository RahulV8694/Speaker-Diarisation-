from pydub import AudioSegment, effects
import os

def enhance_audio(audio_path):
    print("Loading audio file...")
    audio = AudioSegment.from_file(audio_path)
    
    print("Applying audio enhancements for better diarization...")
    
    normalized_audio = effects.normalize(audio)
    
    filtered_audio = normalized_audio.high_pass_filter(80)
    
    filtered_audio = filtered_audio.low_pass_filter(8000)
    
    filtered_audio = filtered_audio + 3
    
    filtered_audio = effects.compress_dynamic_range(filtered_audio, threshold=-20, ratio=4)
    
    filtered_audio = effects.normalize(filtered_audio)
    
    enhanced_path = os.path.splitext(audio_path)[0] + "_enhanced.wav"
    print(f"Saving enhanced audio to: {enhanced_path}")
    filtered_audio.export(enhanced_path, format="wav", parameters=["-ar", "16000"])
    
    return enhanced_path