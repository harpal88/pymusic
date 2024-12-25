import numpy as np
import librosa

def get_peak_frequencies(filename, n_fft):
    y, sr = librosa.load(filename)
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    frequencies = librosa.fft_frequencies(sr=sr)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units='frames', delta=0.01)

    peak_times = []
    peak_freqs = []
    peak_amplitudes = []

    for onset_frame in onset_frames:
        if onset_frame < S.shape[1]:
            spectrum = np.abs(S[:, onset_frame])
            peak = np.argmax(spectrum)
            peak_freq = frequencies[peak]
            peak_amp = np.max(spectrum)
            if 50 <= peak_freq <= 1000:
                peak_time = librosa.frames_to_time(onset_frame, sr=sr)
                peak_times.append(peak_time)
                peak_freqs.append(peak_freq)
                peak_amplitudes.append(peak_amp)

    return np.array(peak_times), np.array(peak_freqs), np.array(peak_amplitudes)

def calculate_pitch_score(detected_freqs, expected_freqs, threshold=0.1, penalize_missing=False):
    matched = 0
    unmatched = 0

    for expected in expected_freqs:
        if len(detected_freqs) == 0:
            unmatched += 1
        else:
            closest_match = min(detected_freqs, key=lambda x: abs(x - expected))
            if abs(closest_match - expected) <= threshold:
                matched += 1
            else:
                unmatched += 1

    total = len(expected_freqs)
    
    if penalize_missing:
        score = (matched / total) * 10
    else:
        score = (matched / (matched + unmatched)) * 10 if (matched + unmatched) > 0 else 10

    return score, matched, unmatched
def calculate_dynamics_score(amps1, amps2, amp_tolerance=0.29, penalize_missing=False):
    matched = 0
    unmatched = 0

    for amp1 in amps1:
        if len(amps2) == 0:
            unmatched += 1
        else:
            closest_amp2 = min(amps2, key=lambda x: abs(x - amp1))
            if abs(closest_amp2 - amp1) <= amp_tolerance:
                matched += 1
            else:
                unmatched += 1

    total = len(amps1)
    
    if penalize_missing:
        score = (matched / total) * 10
    else:
        score = (matched / (matched + unmatched)) * 10 if (matched + unmatched) > 0 else 10
    
    return score, matched
    

def calculate_time_accuracy(times1, times2, time_tolerance=0.15, penalize_missing=False):
    matched = 0
    unmatched = 0

    for time1 in times1:
        if len(times2) == 0:
            unmatched += 1
        else:
            closest_time2 = min(times2, key=lambda x: abs(x - time1))
            if abs(closest_time2 - time1) <= time_tolerance:
                matched += 1
            else:
                unmatched += 1

    total = len(times1)
    
    if penalize_missing:
        score = (matched / total) * 10
    else:
        score = (matched / (matched + unmatched)) * 10 if (matched + unmatched) > 0 else 10

    return score


def calculate_rhythm_accuracy(times1, times2, grid_tolerance=0.08, penalize_missing=False):
    total_intervals = len(times1) - 1
    matched = 0
    unmatched = 0

    if total_intervals > 0:
        expected_intervals = np.diff(times1)
        detected_intervals = np.diff(times2)

        for exp_interval in expected_intervals:
            if len(detected_intervals) == 0:
                unmatched += 1
            else:
                closest_det_interval = min(detected_intervals, key=lambda x: abs(x - exp_interval))
                if abs(closest_det_interval - exp_interval) <= grid_tolerance:
                    matched += 1
                else:
                    unmatched += 1
    
    if penalize_missing:
        score = (matched / total_intervals) * 10 if total_intervals > 0 else 10
    else:
        score = (matched / (matched + unmatched)) * 10 if (matched + unmatched) > 0 else 10

    return score


def adjust_pitch_based_on_dynamics(pitch_score, dynamics_score):
    if dynamics_score < 10:
        adjustment_factor = (10 - dynamics_score) * 0.1
        pitch_score = min(10, pitch_score + adjustment_factor)
    return pitch_score

def adjust_dynamics_based_on_time(dynamics_score, time_accuracy):
    if time_accuracy < 10:
        adjustment_factor = (10 - time_accuracy) * 1
        dynamics_score = min(10, dynamics_score + adjustment_factor)
    return dynamics_score

def compare_n_fft(filename1, filename2):
    n_fft_values = [512, 1024, 2048]
    total_pitch_score = 0
    total_dynamics_score = 0
    total_time_accuracy = 0
    total_rhythm_accuracy = 0
    num_fft = len(n_fft_values)

    for n_fft in n_fft_values:
        times1, freqs1, amps1 = get_peak_frequencies(filename1, n_fft)
        times2, freqs2, amps2 = get_peak_frequencies(filename2, n_fft)

        pitch_score, _, _ = calculate_pitch_score(freqs2, freqs1, penalize_missing=False)
        dynamics_score, _ = calculate_dynamics_score(amps1, amps2, penalize_missing=False)
        time_accuracy = calculate_time_accuracy(times1, times2, penalize_missing=False)
        rhythm_accuracy = calculate_rhythm_accuracy(times1, times2, penalize_missing=False)

        adjusted_dynamics_score = adjust_dynamics_based_on_time(dynamics_score, time_accuracy)
        adjusted_pitch_score = adjust_pitch_based_on_dynamics(pitch_score, adjusted_dynamics_score)

        total_pitch_score += adjusted_pitch_score
        total_dynamics_score += adjusted_dynamics_score
        total_time_accuracy += time_accuracy
        total_rhythm_accuracy += rhythm_accuracy

    avg_pitch_score = total_pitch_score / num_fft
    avg_dynamics_score = total_dynamics_score / num_fft
    avg_time_accuracy = total_time_accuracy / num_fft
    avg_rhythm_accuracy = total_rhythm_accuracy / num_fft

    print(f"Average Pitch Score: {avg_pitch_score:.2f}/10")
    print(f"Average Dynamics Score: {avg_dynamics_score:.2f}/10")
    print(f"Average Time Accuracy: {avg_time_accuracy:.2f}/10")
    print(f"Average Rhythm Accuracy: {avg_rhythm_accuracy:.2f}/10")


# Example usage
filename1 = 'Original_32_notes.wav'
filename2 = '5_notes_off_pitch_5_high_v (2).wav'  # Rep///////.......................................................................lace with the actual second file

compare_n_fft(filename1, filename2)
