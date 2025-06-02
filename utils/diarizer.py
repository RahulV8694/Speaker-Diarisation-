from pyannote.audio import Pipeline
import matplotlib.pyplot as plt
import os

def run_diarization(audio_path: str, hf_token: str):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
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
