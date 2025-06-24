from pydub import AudioSegment, effects
import os

def enhance_audio(audio_path):

    audio = AudioSegment.from_file(audio_path)

    normalized_audio = effects.normalize(audio)

    filtered_audio = normalized_audio.high_pass_filter(100)

    enhanced_path = os.path.splitext(audio_path)[0] + "_enhanced.wav"
    filtered_audio.export(enhanced_path, format="wav")
    return enhanced_path