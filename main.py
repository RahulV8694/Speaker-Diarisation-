import os
from utils.audio_extractor import extract_audio
from utils.diarizer import run_diarization, print_segments, plot_diarization

# === USER INPUT ===
VIDEO_PATH = "data/input_video.mp4"  # Replace with your actual file
HUGGINGFACE_TOKEN = "your_hf_token_here"  # Add your HF token

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        exit(1)

    print("🔁 Extracting audio...")
    audio_path = extract_audio(VIDEO_PATH)

    print("🧠 Running speaker diarization...")
    diarization = run_diarization(audio_path, HUGGINGFACE_TOKEN)

    print("\n🗣️ Speaker Segments:")
    print_segments(diarization)

    print("📊 Plotting timeline...")
    plot_diarization(diarization)
