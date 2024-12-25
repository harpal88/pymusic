import numpy as np
import librosa

def calculate_timing_accuracy(original_file, user_file):
    def detect_onsets(audio_file):
        y, sr = librosa.load(audio_file)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        return onsets
    
    def calculate_timing_score(original_onsets, user_onsets):
        if original_onsets.size == 0 or user_onsets.size == 0:
            return 0.0

        original_onsets = np.sort(original_onsets)
        user_onsets = np.sort(user_onsets)
        
        original_diffs = np.diff(original_onsets)
        tempo = np.average(original_diffs) if len(original_diffs) > 0 else 1
        
        threshold = tempo * 0.19  # Relaxed threshold based on tempo
        
        correct_timing = 0
        
        for orig_onset in original_onsets:
            if any(abs(orig_onset - user_onset) <= threshold for user_onset in user_onsets):
                correct_timing += 1
        
        total_onsets = len(original_onsets)
        if total_onsets == 0:
            return 0.0
        
        accuracy_ratio = correct_timing / total_onsets
        timing_accuracy = round(accuracy_ratio * 10, 1)  # Adjusted scoring
        
        timing_accuracy = min(timing_accuracy, 10.0)  # Cap the score at 10
        
        return timing_accuracy
    
    original_onsets = detect_onsets(original_file)
    user_onsets = detect_onsets(user_file)
    
    timing_accuracy = calculate_timing_score(np.array(original_onsets), np.array(user_onsets))
    
    return {"timing_accuracy": timing_accuracy, "user_onsets": user_onsets, "original_onsets": original_onsets}

def adjust_onsets_for_timing(original_onsets, user_onsets):
    adjusted_user_onsets = np.copy(user_onsets)
    
    for i, user_onset in enumerate(adjusted_user_onsets):
        closest_original = original_onsets[np.argmin(np.abs(original_onsets - user_onset))]
        adjusted_user_onsets[i] = closest_original
        
    return adjusted_user_onsets

def calculate_pitch_accuracy(original_file, user_file, adjusted_user_onsets=None, tolerance=0.00015):
    def extract_pitches(audio_file, onsets=None):
        y, sr = librosa.load(audio_file)
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        return np.array(pitch_values)
    
    def calculate_pitch_score(original_pitches, user_pitches):
        if original_pitches.size == 0 or user_pitches.size == 0:
            return 0.0

        correct_pitches = 0
        for orig_pitch in original_pitches:
            if any(abs(librosa.hz_to_midi(orig_pitch) - librosa.hz_to_midi(user_pitch)) <= tolerance for user_pitch in user_pitches):
                correct_pitches += 1
        
        total_pitches = len(original_pitches)
        if total_pitches == 0:
            return 0.0
        
        accuracy_ratio = correct_pitches / total_pitches
        pitch_accuracy = round(accuracy_ratio * 10, 2)
        
        return pitch_accuracy
    
    original_pitches = extract_pitches(original_file)
    
    if adjusted_user_onsets is not None:
        user_pitches = extract_pitches(user_file, onsets=adjusted_user_onsets)
    else:
        user_pitches = extract_pitches(user_file)
    
    pitch_accuracy = calculate_pitch_score(original_pitches, user_pitches)
    
    return {"pitch_accuracy": pitch_accuracy}

def calculate_adjusted_pitch_accuracy(original_file, user_file):
    timing_result = calculate_timing_accuracy(original_file, user_file)
    timing_accuracy = timing_result["timing_accuracy"]
    
    if timing_accuracy < 10.0:
        adjusted_user_onsets = adjust_onsets_for_timing(timing_result["original_onsets"], timing_result["user_onsets"])
        pitch_result = calculate_pitch_accuracy(original_file, user_file, adjusted_user_onsets=adjusted_user_onsets)
    else:
        pitch_result = calculate_pitch_accuracy(original_file, user_file)
    
    return {
        "timing_accuracy": timing_accuracy,
        "pitch_accuracy": pitch_result["pitch_accuracy"]
    }

def process_all_files(original_file, user_files):
    results = []
    for user_file in user_files:
        result = calculate_adjusted_pitch_accuracy(original_file, user_file)
        results.append({"file": user_file, "timing_accuracy": result["timing_accuracy"], "pitch_accuracy": result["pitch_accuracy"]})
    return results

# List of user files to test
user_files = [
    "3_extra_notes_2_off_pitch_1_off_timing (2).wav",
    "5_notes_high_volume.wav",
    "5_notes_missing (1).wav",
    "5_notes_off_pitch (3).wav",
    "5_notes_off_pitch_5_high_v (2).wav",
    "5_notes_off_time.wav",
    "7.wav",
    "8_notes_longer_hold (1).wav",
    "8.5.wav",
    "9.5.wav",
    "10_notes_high_tempo.wav",
    "10_notes_high_tempo0_notes_low_tempo.wav",
    "10_notes_low_volume.wav",
    "10_notes_mixed_volume.wav",
    "10_notes_off_time.wav",
    "10_notes_staccato (2).wav"
]

# Path to the original reference file
original_file = 'Original_32_notes.wav'

# Process all files
all_results = process_all_files(original_file, user_files)

# Print the results
for result in all_results:
    print(f"File: {result['file']}, Timing Accuracy: {result['timing_accuracy']}/10, Pitch Accuracy: {result['pitch_accuracy']}/10")
