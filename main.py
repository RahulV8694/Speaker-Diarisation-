import os
from utils.audio_extractor import extract_audio
from utils.audio_enhancer import enhance_audio  
from utils.diarizer import run_diarization, print_segments, plot_diarization
from utils.quality_assessor import assess_diarization_quality, suggest_improvements

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

    print("=== Enhanced Speaker Diarization with Quality Assessment ===")
    print(f"Expected speakers: {MIN_SPEAKERS}-{MAX_SPEAKERS}")
    print()

    print("1. Extracting audio...")
    audio_path = extract_audio(VIDEO_PATH)

    print("2. Applying basic audio enhancements...")
    enhanced_audio_path = enhance_audio(audio_path)  

    print("3. Running speaker diarization with optimized parameters...")
    diarization = run_diarization(
        audio_path,
        HUGGINGFACE_TOKEN,
        min_speakers=MIN_SPEAKERS,
        max_speakers=MAX_SPEAKERS
    )

    print("\n4. Speaker Segments:")
    print_segments(diarization)

    print("\n5. Assessing diarization quality...")
    quality_metrics = assess_diarization_quality(diarization, audio_path)
    
    print("\n6. Quality improvement suggestions:")
    suggestions = suggest_improvements(quality_metrics)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")

    print("\n7. Creating visualization...")
    plot_diarization(diarization)
    
    print("\n=== Enhanced Diarization Complete ===")
    print(f"Quality Score: {quality_metrics['quality_score']:.1f}/100")
