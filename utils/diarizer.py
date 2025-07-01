from pyannote.audio import Pipeline
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import librosa

def run_diarization(audio_path: str, hf_token: str, min_speakers=None, max_speakers=None):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU acceleration available)")
    
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
    
    pipeline.to(device)
    
    print(f"Processing audio with device: {device}")
    
    audio_duration, audio_characteristics = analyze_audio_characteristics(audio_path)
    print(f"Audio duration: {audio_duration:.1f}s")
    print(f"Audio characteristics: {audio_characteristics}")
    
    if min_speakers is None or max_speakers is None:
        min_speakers, max_speakers = estimate_speaker_count(audio_duration, audio_characteristics)
        print(f"Estimated speakers: {min_speakers}-{max_speakers}")
    
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    
    print(f"Diarization parameters: {kwargs}")
    
    try:
        diarization = pipeline(audio_path, **kwargs)
        return diarization
    except Exception as e:
        print(f"Error during diarization: {e}")
        print("Trying with fallback parameters...")
        
        fallback_kwargs = {
            "min_speakers": max(1, min_speakers - 1) if min_speakers else 1,
            "max_speakers": min(5, max_speakers + 1) if max_speakers else 5
        }
        print(f"Fallback parameters: {fallback_kwargs}")
        return pipeline(audio_path, **fallback_kwargs)

def analyze_audio_characteristics(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        
        rms = np.sqrt(np.mean(y**2))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        vad = librosa.effects.preemphasis(y)
        vad_energy = np.sum(vad**2) / len(vad)
        
        characteristics = {
            "rms": rms,
            "zero_crossing_rate": zero_crossing_rate,
            "spectral_centroid": spectral_centroid,
            "vad_energy": vad_energy,
            "has_voice": vad_energy > 0.001
        }
        
        return duration, characteristics
        
    except Exception as e:
        print(f"Warning: Could not analyze audio characteristics: {e}")
        return 0, {"has_voice": True}

def estimate_speaker_count(duration, characteristics):
    if duration < 30:
        return 1, 2
    elif duration < 120:
        return 1, 3
    elif duration < 600:
        return 2, 4
    else:
        return 2, 6

def print_segments(diarization):
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"{turn.start:.2f}s --> {turn.end:.2f}s : {speaker}")

def plot_diarization(diarization, output_path="outputs/diarization_plot.png"):
    fig, ax = plt.subplots(figsize=(24, 10))
    
    segments = []
    speakers = set()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
        speakers.add(speaker)
    
    speakers = sorted(list(speakers))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    speaker_colors = {speaker: colors[i % len(colors)] for i, speaker in enumerate(speakers)}
    
    y_positions = {speaker: i for i, speaker in enumerate(speakers)}
    
    for start, end, speaker in segments:
        y_pos = y_positions[speaker]
        color = speaker_colors[speaker]
        
        duration = end - start
        height = 0.8
        
        rect = plt.Rectangle((start, y_pos - height/2), duration, height, 
                           facecolor=color, alpha=0.9, edgecolor='white', 
                           linewidth=2, zorder=3)
        ax.add_patch(rect)
    
    ax.set_xlabel('Time (seconds)', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.set_ylabel('Speakers', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.set_title('Audio Speaker Diarization Timeline', fontsize=20, fontweight='bold', 
                pad=30, color='#2c3e50')
    
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#34495e')
    
    if len(speakers) > 0:
        ax.set_ylim(-0.5, len(speakers) - 0.5)
        ax.set_yticks(range(len(speakers)))
        ax.set_yticklabels([f'Speaker {i+1}' for i in range(len(speakers))], 
                          fontsize=12, fontweight='bold')
    else:
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
    
    if segments:
        max_time = max([end for _, end, _ in segments])
        time_marks = list(range(0, int(max_time) + 60, 60))
        ax.set_xticks(time_marks)
        ax.set_xticklabels([f'{t//60}:{t%60:02d}' for t in time_marks], 
                          fontsize=11, fontweight='bold')
    else:
        ax.set_xticks([0])
        ax.set_xticklabels(['0:00'], fontsize=11, fontweight='bold')
    
    if speakers:
        legend_elements = []
        for speaker in speakers:
            legend_elements.append(plt.Rectangle((0,0), 1, 1, 
                                               facecolor=speaker_colors[speaker], 
                                               alpha=0.9, edgecolor='white', linewidth=2))
        
        legend = ax.legend(legend_elements, [f'Speaker {i+1}' for i in range(len(speakers))], 
                          loc='upper right', title='Speaker Legend', 
                          title_fontsize=14, fontsize=12,
                          frameon=True, fancybox=True, shadow=True, 
                          framealpha=0.95, edgecolor='#34495e')
        legend.get_frame().set_facecolor('#ecf0f1')
    
    if segments:
        total_duration = max([end for _, end, _ in segments])
        num_segments = len(segments)
        avg_segment_duration = total_duration / num_segments if num_segments > 0 else 0
        
        stats_text = (f'Total Duration: {total_duration//60:.0f}m {total_duration%60:.0f}s | '
                     f'Segments: {num_segments} | Speakers: {len(speakers)} | '
                     f'Avg Segment: {avg_segment_duration:.1f}s')
    else:
        stats_text = f'No speakers detected | Segments: 0 | Speakers: 0'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', 
                     alpha=0.9, edgecolor='#34495e', linewidth=2))
    
    if segments and speakers:
        for i, speaker in enumerate(speakers):
            speaker_segments = [seg for seg in segments if seg[2] == speaker]
            total_speaker_time = sum(end - start for start, end, _ in speaker_segments)
            percentage = (total_speaker_time / total_duration) * 100 if total_duration > 0 else 0
            
            ax.text(0.98, i, f'{percentage:.1f}%', transform=ax.transAxes, 
                   fontsize=10, fontweight='bold', ha='right', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=speaker_colors[speaker], 
                            alpha=0.8, edgecolor='white', linewidth=1))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#34495e')
    ax.spines['bottom'].set_color('#34495e')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='#f8f9fa', edgecolor='none')
    print(f"Saved enhanced audio visualization to {output_path}")
    plt.show()
