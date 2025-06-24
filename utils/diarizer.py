from pyannote.audio import Pipeline
import matplotlib.pyplot as plt
import os
import torch

def run_diarization(audio_path: str, hf_token: str):
    # Check for GPU availability and set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU acceleration available)")
    
    # Load pipeline with GPU acceleration
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", 
        use_auth_token=hf_token
    )
    
    # Move pipeline to GPU if available
    pipeline.to(device)
    
    print(f"Processing audio with device: {device}")
    diarization = pipeline(audio_path)
    return diarization

def print_segments(diarization):
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"{turn.start:.2f}s --> {turn.end:.2f}s : {speaker}")

def plot_diarization(diarization, output_path="outputs/diarization_plot.png"):
    fig = diarization.plot()
    plt.title("Speaker Diarization Timeline")
    plt.xlabel("Time (s)")
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
