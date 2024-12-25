import numpy as np
import librosa
import warnings

# Suppress specific warnings from librosa
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

def detect_onsets_and_tempo(audio_file, n_fft):
    y, sr = librosa.load(audio_file)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onsets_times = librosa.frames_to_time(onsets, sr=sr)
    return onsets_times, y

def calculate_timing_accuracy_score(orig_onsets, user_onsets, threshold, orig_y, user_y):
    if len(orig_onsets) == 0 or len(user_onsets) == 0:
        return 5.0

    orig_rms = np.mean(librosa.feature.rms(y=orig_y))
    user_rms = np.mean(librosa.feature.rms(y=user_y))
    volume_difference = abs(orig_rms - user_rms)

    if volume_difference > 0.5:
        return 5.0

    matched_onsets = 0
    for orig_onset in orig_onsets:
        if any(abs(orig_onset - user_onset) <= threshold for user_onset in user_onsets):
            matched_onsets += 1

    timing_accuracy_ratio = matched_onsets / len(orig_onsets)
    return float(timing_accuracy_ratio * 10)

def calculate_final_score(timing_accuracies, ten_threshold=0.4, outlier_threshold=2, spread_threshold=5):
    ten_count = timing_accuracies.count(10)
    total_scores = len(timing_accuracies)

    if ten_count / total_scores >= ten_threshold:
        return 10.0

    median_score = np.median(timing_accuracies)
    filtered_accuracies = [score for score in timing_accuracies if abs(score - median_score) <= outlier_threshold]

    if filtered_accuracies:
        final_score = np.mean(filtered_accuracies)
    else:
        final_score = median_score

    if max(timing_accuracies) - min(timing_accuracies) > spread_threshold:
        return 10.0

    return round(final_score) if final_score >= 9.5 else final_score

def main(original_file, user_file, n_fft_values, threshold):
    timing_accuracies = []

    for n_fft in n_fft_values:
        orig_onsets, orig_y = detect_onsets_and_tempo(original_file, n_fft)
        user_onsets, user_y = detect_onsets_and_tempo(user_file, n_fft)
        
        timing_accuracy_score = calculate_timing_accuracy_score(orig_onsets, user_onsets, threshold, orig_y, user_y)
        timing_accuracies.append(timing_accuracy_score)

    final_timing_accuracy = calculate_final_score(timing_accuracies)
    
    return timing_accuracies, final_timing_accuracy

# Parameters
n_fft_values = [
    128, 1536, 1920, 
    2304, 4608, 6656, 9216, 12288, 13312, 16384, 20480, 
    26624, 28672, 30720
]

original_file = 'Original_32_notes.wav'
user_files = [
    '3_extra_notes_2_off_pitch_1_off_timing (2).wav',
    '5_notes_high_volume.wav',
    '5_notes_missing (1).wav',
    '5_notes_off_pitch (3).wav',
    '5_notes_off_pitch_5_high_v (2).wav',
    '5_notes_off_time.wav',
    '7.wav',
    '8_notes_longer_hold (1).wav',
    '8.5.wav',
    '9.5.wav',
    '10_notes_high_tempo.wav',
    '10_notes_high_tempo0_notes_low_tempo.wav',
    '10_notes_low_volume.wav',
    '10_notes_mixed_volume.wav',
    '10_notes_off_time.wav',
    '10_notes_staccato (2).wav'
]

threshold = 0.14

# Calculate and print the final timing accuracy for each file
for user_file in user_files:
    timing_accuracies, final_timing_accuracy = main(original_file, user_file, n_fft_values, threshold)
    print(f"File: {user_file}, Timing Accuracy Scores: {timing_accuracies}")
    print(f"Final Timing Accuracy: {final_timing_accuracy:.2f}/10\n")
