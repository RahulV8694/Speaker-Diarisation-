from pyannote.audio import Pipeline
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

def run_diarization(audio_path: str, hf_token: str, min_speakers=None, max_speakers=None):
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
    
    # Load pipeline with optimized local config for better quality
    config_path = "local_config.yaml"
    if os.path.exists(config_path):
        print(f"Loading optimized pipeline from: {config_path}")
        pipeline = Pipeline.from_pretrained(
            config_path,
            use_auth_token=hf_token
        )
    else:
        print("Using default pipeline (local config not found)")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization", 
            use_auth_token=hf_token
        )
    
    # Move pipeline to GPU if available
    pipeline.to(device)
    
    print(f"Processing audio with device: {device}")
    
    # Only pass min_speakers and max_speakers
    diarization_params = {
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }
    diarization_params = {k: v for k, v in diarization_params.items() if v is not None}
    print(f"Diarization parameters: {diarization_params}")
    diarization = pipeline(audio_path, **diarization_params)
    return diarization

def print_segments(diarization):
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"{turn.start:.2f}s --> {turn.end:.2f}s : {speaker}")

def plot_diarization(diarization, output_path="outputs/diarization_plot.png"):

    fig, ax = plt.subplots(figsize=(15, 6))

    segments = []
    speakers = set()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
        speakers.add(speaker)
    
    # Create color map for speakers
    colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
    speaker_colors = {speaker: colors[i] for i, speaker in enumerate(speakers)}
    

    y_pos = 0
    for start, end, speaker in segments:
        color = speaker_colors[speaker]
        ax.barh(y_pos, end - start, left=start, height=0.8, 
                color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Speaker Segments')
    ax.set_title('Speaker Diarization Timeline')
    ax.grid(True, alpha=0.3)
    

    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    

    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=speaker_colors[speaker], 
                                   alpha=0.7, edgecolor='black') for speaker in speakers]
    ax.legend(legend_elements, speakers, loc='upper right', title='Speakers')
    

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.show()
