import librosa
import numpy as np

# Function to estimate tempo with enhanced onset detection
def estimate_tempo(file_path, ref_tempo=None):
    y, sr = librosa.load(file_path)
    
    # Enhanced onset detection with tuned parameters
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    onset_env = np.convolve(onset_env, np.ones(10)/10, mode='same')  # Smoothing
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # Dynamic tempo adjustment
    if ref_tempo is not None:
        if tempo > ref_tempo * 1.3:
            tempo /= 2
        elif tempo < ref_tempo / 1.3:
            tempo *= 2
    
    return float(tempo)

# List of files to compare
files_to_compare = [
    'Original_32_notes.wav',
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

# Estimate the reference tempo from the original file
reference_file = files_to_compare[0]
ref_tempo = estimate_tempo(reference_file)

# Store results
results = []

# Compare each file with the reference tempo
for file in files_to_compare[1:]:  # Skip the reference file
    test_tempo = estimate_tempo(file, ref_tempo)
    
    # Calculate tempo accuracy as a percentage of deviation
    tempo_accuracy = 100 - abs((test_tempo - ref_tempo) / ref_tempo * 100)
    
    # Apply a threshold for significant differences
    significant_difference = abs(test_tempo - ref_tempo) > 5  # Example threshold of 5 BPM
    
    results.append({
        'file': file,
        'ref_tempo': ref_tempo,
        'test_tempo': test_tempo,
        'tempo_accuracy': tempo_accuracy,
        'significant_difference': significant_difference
    })

# Display results
for result in results:
    print(f"File: {result['file']}")
    print(f"Reference Tempo: {result['ref_tempo']} BPM")
    print(f"Test Tempo: {result['test_tempo']} BPM")
    print(f"Tempo Accuracy: {result['tempo_accuracy']:.2f}%")
    if result['significant_difference']:
        print("Significant tempo difference detected!")
    print("-" * 50)
