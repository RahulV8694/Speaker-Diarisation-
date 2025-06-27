import os
import glob
from utils.audio_extractor import extract_audio
from utils.audio_enhancer import enhance_audio  
from utils.diarizer import run_diarization, print_segments, plot_diarization, write_segments_to_file

# === USER INPUT ===
# VIDEO_PATH = "C:\\Users\\odink\\Documents\\Adobe\\Premiere Pro\\25.0\\028_0001_client_w_server_audio.mp4"  # Replace with your actual file
HUGGINGFACE_TOKEN = ""  # Add your HF token

# Speaker Pre setup
MIN_SPEAKERS = 2  
MAX_SPEAKERS = 2  

if __name__ == "__main__":
    folder_path = 'C:\\Users\\odink\\Documents\\Adobe\\Premiere Pro\\25.0'
    mp4_files = glob.glob(os.path.join(folder_path, '*.mp4'))
    
    for video_path in mp4_files:
        # print(video_path)
        file_name_wo_ext = os.path.splitext(os.path.basename(video_path))[0]
        print(file_name_wo_ext)
        print("=== Speaker Diarization with Enhanced Quality ===")
        print(f"Expected speakers: {MIN_SPEAKERS}-{MAX_SPEAKERS}")
        print()

        print("1. Extracting audio...")
        audio_path = extract_audio(video_path)

        print("2. Enhancing audio quality...")
        enhanced_audio_path = enhance_audio(audio_path)  

        print("3. Running speaker diarization with optimized parameters...")
        diarization = run_diarization(
            enhanced_audio_path, 
            HUGGINGFACE_TOKEN,
            min_speakers=MIN_SPEAKERS,
            max_speakers=MAX_SPEAKERS
        )

        if not os.path.exists(f".\\segmentations\\{file_name_wo_ext}.txt"):
            write_segments_to_file(diarization, f".\\segmentations\\{file_name_wo_ext}.txt")
        else:
            print(f"File Already Exists: .\\segmentations\\{file_name_wo_ext}.txt")
