import numpy as np
import librosa
from typing import Dict, List, Tuple

def assess_diarization_quality(diarization, audio_path: str) -> Dict:
    print("Assessing diarization quality...")
    
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    segments = []
    speakers = set()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
        speakers.add(speaker)
    
    metrics = {}
    
    coverage = calculate_coverage(segments, duration)
    metrics['coverage'] = coverage
    
    segment_stats = analyze_segment_lengths(segments)
    metrics['segment_stats'] = segment_stats
    
    speaker_distribution = analyze_speaker_distribution(segments, speakers)
    metrics['speaker_distribution'] = speaker_distribution
    
    silence_analysis = analyze_silence(segments, duration)
    metrics['silence_analysis'] = silence_analysis
    
    quality_score = calculate_quality_score(metrics)
    metrics['quality_score'] = quality_score
    
    print_quality_report(metrics, len(speakers))
    
    return metrics

def calculate_coverage(segments: List[Tuple], duration: float) -> Dict:
    total_covered = sum(end - start for start, end, _ in segments)
    coverage_percentage = (total_covered / duration) * 100
    
    return {
        'total_covered': total_covered,
        'total_duration': duration,
        'coverage_percentage': coverage_percentage,
        'uncovered_duration': duration - total_covered
    }

def analyze_segment_lengths(segments: List[Tuple]) -> Dict:
    lengths = [end - start for start, end, _ in segments]
    
    if not lengths:
        return {
            'count': 0,
            'mean_length': 0.0,
            'median_length': 0.0,
            'min_length': 0.0,
            'max_length': 0.0,
            'std_length': 0.0,
            'short_segments': 0,
            'long_segments': 0
        }
    
    return {
        'count': len(lengths),
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'std_length': np.std(lengths),
        'short_segments': len([l for l in lengths if l < 1.0]),
        'long_segments': len([l for l in lengths if l > 30.0])
    }

def analyze_speaker_distribution(segments: List[Tuple], speakers: set) -> Dict:
    speaker_times = {}
    for speaker in speakers:
        speaker_segments = [seg for seg in segments if seg[2] == speaker]
        total_time = sum(end - start for start, end, _ in speaker_segments)
        speaker_times[speaker] = total_time
    
    total_time = sum(speaker_times.values())
    
    if total_time == 0:
        speaker_percentages = {speaker: 0.0 for speaker in speakers}
    else:
        speaker_percentages = {speaker: (time/total_time)*100 for speaker, time in speaker_times.items()}
    
    return {
        'speaker_times': speaker_times,
        'speaker_percentages': speaker_percentages,
        'total_time': total_time,
        'balance_score': calculate_balance_score(speaker_percentages)
    }

def analyze_silence(segments: List[Tuple], duration: float) -> Dict:
    sorted_segments = sorted(segments, key=lambda x: x[0])
    
    silence_periods = []
    current_time = 0
    
    for start, end, _ in sorted_segments:
        if start > current_time:
            silence_periods.append((current_time, start))
        current_time = max(current_time, end)
    
    if current_time < duration:
        silence_periods.append((current_time, duration))
    
    silence_durations = [end - start for start, end in silence_periods]
    
    return {
        'silence_periods': silence_periods,
        'total_silence': sum(silence_durations),
        'silence_percentage': (sum(silence_durations) / duration) * 100,
        'num_silence_periods': len(silence_periods),
        'avg_silence_duration': np.mean(silence_durations) if silence_durations else 0,
        'long_silences': len([s for s in silence_durations if s > 5.0])
    }

def calculate_balance_score(speaker_percentages: Dict) -> float:
    if len(speaker_percentages) <= 1:
        return 1.0
    
    percentages = list(speaker_percentages.values())
    mean_percentage = np.mean(percentages)
    variance = np.var(percentages)
    
    max_variance = (100 / len(percentages)) ** 2
    balance_score = 1 - (variance / max_variance)
    
    return max(0, min(1, balance_score))

def calculate_quality_score(metrics: Dict) -> float:
    score = 0
    
    coverage = metrics['coverage']['coverage_percentage']
    if coverage > 80:
        score += 30
    elif coverage > 60:
        score += 20
    elif coverage > 40:
        score += 10
    
    segment_stats = metrics['segment_stats']
    if segment_stats['count'] == 0:
        return 0.0
    
    short_segments_ratio = segment_stats['short_segments'] / segment_stats['count']
    if short_segments_ratio < 0.1:
        score += 25
    elif short_segments_ratio < 0.2:
        score += 20
    elif short_segments_ratio < 0.3:
        score += 15
    
    balance_score = metrics['speaker_distribution']['balance_score']
    score += balance_score * 25
    
    silence_analysis = metrics['silence_analysis']
    silence_percentage = silence_analysis['silence_percentage']
    if 10 < silence_percentage < 40:
        score += 20
    elif 5 < silence_percentage < 50:
        score += 15
    elif 0 < silence_percentage < 60:
        score += 10
    
    return min(100, score)

def print_quality_report(metrics: Dict, num_speakers: int):
    print("\n" + "="*60)
    print("DIARIZATION QUALITY ASSESSMENT")
    print("="*60)
    
    quality_score = metrics['quality_score']
    print(f"\n📊 Overall Quality Score: {quality_score:.1f}/100")
    
    if quality_score >= 80:
        print("🎉 Excellent diarization quality!")
    elif quality_score >= 60:
        print("✅ Good diarization quality")
    elif quality_score >= 40:
        print("⚠️  Fair diarization quality - some improvements possible")
    else:
        print("❌ Poor diarization quality - significant improvements needed")
    
    coverage = metrics['coverage']
    print(f"\n📈 Coverage Analysis:")
    print(f"   • Audio covered: {coverage['coverage_percentage']:.1f}%")
    print(f"   • Uncovered time: {coverage['uncovered_duration']:.1f}s")
    
    segment_stats = metrics['segment_stats']
    print(f"\n⏱️  Segment Analysis:")
    print(f"   • Total segments: {segment_stats['count']}")
    print(f"   • Average length: {segment_stats['mean_length']:.1f}s")
    print(f"   • Short segments (<1s): {segment_stats['short_segments']}")
    print(f"   • Long segments (>30s): {segment_stats['long_segments']}")
    
    speaker_dist = metrics['speaker_distribution']
    print(f"\n👥 Speaker Distribution ({num_speakers} speakers):")
    for speaker, percentage in speaker_dist['speaker_percentages'].items():
        print(f"   • {speaker}: {percentage:.1f}%")
    print(f"   • Balance score: {speaker_dist['balance_score']:.2f}")
    
    silence = metrics['silence_analysis']
    print(f"\n🔇 Silence Analysis:")
    print(f"   • Silence percentage: {silence['silence_percentage']:.1f}%")
    print(f"   • Silence periods: {silence['num_silence_periods']}")
    print(f"   • Long silences (>5s): {silence['long_silences']}")
    
    print(f"\n💡 Recommendations:")
    if coverage['coverage_percentage'] < 60:
        print("   • Low coverage detected - consider adjusting segmentation threshold")
    if segment_stats['short_segments'] > segment_stats['count'] * 0.2:
        print("   • Many short segments - consider increasing min_duration_on")
    if speaker_dist['balance_score'] < 0.5:
        print("   • Unbalanced speaker distribution - speakers may be misidentified")
    if silence['silence_percentage'] > 50:
        print("   • High silence percentage - audio may need better preprocessing")
    
    print("="*60)

def suggest_improvements(metrics: Dict) -> List[str]:
    suggestions = []
    
    coverage = metrics['coverage']['coverage_percentage']
    if coverage < 60:
        suggestions.append("Lower segmentation threshold to capture more speech")
    
    segment_stats = metrics['segment_stats']
    if segment_stats['short_segments'] > segment_stats['count'] * 0.2:
        suggestions.append("Increase min_duration_on to reduce fragmentation")
    
    speaker_dist = metrics['speaker_distribution']
    if speaker_dist['balance_score'] < 0.5:
        suggestions.append("Adjust clustering threshold for better speaker separation")
    
    silence = metrics['silence_analysis']
    if silence['silence_percentage'] > 50:
        suggestions.append("Improve audio preprocessing to reduce background noise")
    
    return suggestions 