import numpy as np
import librosa

def calculate_duration_accuracy(original_file, user_file):
    def detect_durations(audio_file):
        y, sr = librosa.load(audio_file)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
        
        # Compute durations between onsets
        durations = np.diff(onsets)
        return durations
    
    def calculate_duration_score(original_durations, user_durations):
        if len(original_durations) == 0 or len(user_durations) == 0:
            return 0.0, []

        # Sort durations for matching
        original_durations = np.sort(original_durations)
        user_durations = np.sort(user_durations)

        # Define threshold for acceptable duration difference
        threshold = 0.2  # Adjust threshold as needed

        correct_durations = 0
        percentage_diffs = []

        user_idx = 0
        for orig_duration in original_durations:
            while user_idx < len(user_durations) and user_durations[user_idx] < orig_duration - threshold:
                user_idx += 1

            if user_idx < len(user_durations) and abs(orig_duration - user_durations[user_idx]) <= threshold:
                correct_durations += 1
                percentage_diffs.append(abs(orig_duration - user_durations[user_idx]) / orig_duration)
                user_idx += 1
            else:
                percentage_diffs.append(1.0)  # No match found, considered completely incorrect

        # Calculate accuracy score based on correct durations
        total_durations = len(original_durations)
        accuracy_ratio = correct_durations / total_durations
        duration_accuracy = round(accuracy_ratio * 10, 1)  # Scale score to 0-10 range
        
        return duration_accuracy, percentage_diffs

    # Detect durations
    original_durations = detect_durations(original_file)
    user_durations = detect_durations(user_file)

    # Calculate duration accuracy score
    duration_accuracy, percentage_diffs = calculate_duration_score(np.array(original_durations), np.array(user_durations))
    
    return {
        "duration_accuracy": duration_accuracy,
        "percentage_diffs": np.array(percentage_diffs)
    }

# Example usage
results = calculate_duration_accuracy('Original_32_notes.wav', '8.5.wav')
print(results)
