import os
from utils.audio_extractor import extract_audio
from utils.audio_enhancer import enhance_audio  
from utils.diarizer import run_diarization, print_segments, plot_diarization

# === USER INPUT ===
VIDEO_PATH = "data/input_video.mp4"  # Replace with your actual file
HUGGINGFACE_TOKEN = "your_hf_token_here"  # Add your HF token

# Speaker Pre setup
MIN_SPEAKERS = 2  
MAX_SPEAKERS = 2  

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        exit(1)

    print("=== Speaker Diarization with Enhanced Quality ===")
    print(f"Expected speakers: {MIN_SPEAKERS}-{MAX_SPEAKERS}")
    print()

    print("1. Extracting audio...")
    audio_path = extract_audio(VIDEO_PATH)

    print("2. Enhancing audio quality...")
    enhanced_audio_path = enhance_audio(audio_path)  

    print("3. Running speaker diarization with optimized parameters...")
    diarization = run_diarization(
        enhanced_audio_path, 
        HUGGINGFACE_TOKEN,
        min_speakers=MIN_SPEAKERS,
        max_speakers=MAX_SPEAKERS
    )

    print("\n4. Speaker Segments:")
    print_segments(diarization)

    print("\n5. Creating visualization...")
    plot_diarization(diarization)
    
    print("\n=== Diarization Complete ===")
