import ffmpeg
import os

def extract_audio(input_video_path: str, output_audio_path: str = "outputs/audio.wav"):
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    ffmpeg.input(input_video_path).output(output_audio_path, ac=1, ar=16000).run(overwrite_output=True)
    return output_audio_path
