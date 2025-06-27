import os
from utils.audio_extractor import extract_audio
from utils.audio_enhancer import enhance_audio  
from utils.diarizer import run_diarization, print_segments, plot_diarization, write_segments_to_file

# === USER INPUT ===
# VIDEO_PATH = "C:\\Users\\odink\\Documents\\Adobe\\Premiere Pro\\25.0\\028_0001_client_w_server_audio.mp4"  # Replace with your actual file
VIDEO_PATH = "C:\\Users\\odink\\Documents\\Adobe\\Premiere Pro\\25.0\\010_client_w_server_audio.mp4"  # Replace with your actual file
HUGGINGFACE_TOKEN = ""  # Add your HF token

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

    write_segments_to_file(diarization, f".\\outputs\\010_segments.txt")

    print("\n5. Creating visualization...")
    plot_diarization(diarization)
    
    print("\n=== Diarization Complete ===")
