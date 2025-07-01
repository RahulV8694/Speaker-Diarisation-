import librosa
import numpy as np
import os

def diagnose_audio(audio_path):
    print(f"Diagnosing audio file: {audio_path}")
    print("="*50)
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Number of samples: {len(y)}")
    
    # Basic statistics
    rms = np.sqrt(np.mean(y**2))
    print(f"RMS energy: {rms:.6f}")
    
    # Check for silence
    silence_threshold = 0.01
    silent_samples = np.sum(np.abs(y) < silence_threshold)
    silence_percentage = (silent_samples / len(y)) * 100
    print(f"Silence percentage: {silence_percentage:.1f}%")
    
    # Check for clipping
    clipped_samples = np.sum(np.abs(y) > 0.95)
    clip_percentage = (clipped_samples / len(y)) * 100
    print(f"Clipped samples: {clip_percentage:.1f}%")
    
    # Spectral analysis
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    print(f"Spectral centroid: {spectral_centroid:.1f} Hz")
    
    # Voice activity detection
    vad = librosa.effects.preemphasis(y)
    vad_energy = np.sum(vad**2) / len(vad)
    print(f"VAD energy: {vad_energy:.6f}")
    
    # Check if audio has meaningful content
    if rms < 0.001:
        print("⚠️  WARNING: Very low RMS energy - audio might be too quiet")
    elif silence_percentage > 80:
        print("⚠️  WARNING: High silence percentage - audio might be mostly silence")
    elif clip_percentage > 5:
        print("⚠️  WARNING: High clipping percentage - audio might be distorted")
    else:
        print("✅ Audio appears to have reasonable characteristics")
    
    return {
        'duration': duration,
        'rms': rms,
        'silence_percentage': silence_percentage,
        'vad_energy': vad_energy
    }

if __name__ == "__main__":
    # Check original audio
    print("ORIGINAL AUDIO:")
    original_stats = diagnose_audio("outputs/audio.wav")
    
    print("\n" + "="*50 + "\n")
    
    # Check enhanced audio
    print("ENHANCED AUDIO:")
    enhanced_stats = diagnose_audio("outputs/audio_enhanced.wav")
    
    print("\n" + "="*50 + "\n")
    
    # Compare
    print("COMPARISON:")
    print(f"RMS change: {enhanced_stats['rms']/original_stats['rms']:.2f}x")
    print(f"Silence change: {enhanced_stats['silence_percentage'] - original_stats['silence_percentage']:.1f}%")
    print(f"VAD energy change: {enhanced_stats['vad_energy']/original_stats['vad_energy']:.2f}x") 