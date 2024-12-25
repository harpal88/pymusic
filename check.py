import numpy as np
import librosa

def extract_pitches_and_dynamics(audio_file, n_fft):
    y, sr = librosa.load(audio_file)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, n_fft=n_fft)
    
    pitch_values = []
    dynamics_values = []
    
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        magnitude = magnitudes[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
            dynamics_values.append(magnitude)
    
    return np.array(pitch_values), np.array(dynamics_values)

def calculate_pitch_score(original_pitches, user_pitches, original_dynamics, user_dynamics, pitch_tolerance=100, dynamics_tolerance_for_pitch=0.1):
    if original_pitches.size == 0 or user_pitches.size == 0:
        return 0.0

    correct_pitches = 0
    user_midi = librosa.hz_to_midi(user_pitches)
    original_midi = librosa.hz_to_midi(original_pitches)

    for orig_midi, orig_dyn in zip(original_midi, original_dynamics):
        if np.any(np.abs(user_midi - orig_midi) <= pitch_tolerance):
            correct_pitches += 1
        else:
            # If pitch is not within tolerance, check if dynamics match with a separate threshold
            matching_dyn_indices = np.where(np.abs(user_dynamics - orig_dyn) <= dynamics_tolerance_for_pitch)[0]
            if len(matching_dyn_indices) > 0:
                correct_pitches += 1
    
    total_pitches = len(original_pitches)
    accuracy_ratio = correct_pitches / total_pitches if total_pitches > 0 else 0
    pitch_accuracy = round(accuracy_ratio * 10, 2)

    return pitch_accuracy

def calculate_dynamics_score(original_dynamics, user_dynamics, original_pitches, user_pitches, dynamics_tolerance=0.1, pitch_tolerance_for_dynamics=100):
    if original_dynamics.size == 0 or user_dynamics.size == 0:
        return 0.0

    correct_dynamics = 0

    for orig_dyn, orig_midi in zip(original_dynamics, original_pitches):
        if np.any(np.abs(user_dynamics - orig_dyn) <= dynamics_tolerance):
            correct_dynamics += 1
        else:
            # If dynamics is not within tolerance, check if pitch matches with a separate threshold
            matching_pitch_indices = np.where(np.abs(librosa.hz_to_midi(user_pitches) - orig_midi) <= pitch_tolerance_for_dynamics)[0]
            if len(matching_pitch_indices) > 0:
                correct_dynamics += 1
    
    total_dynamics = len(original_dynamics)
    accuracy_ratio = correct_dynamics / total_dynamics if total_dynamics > 0 else 0
    dynamics_accuracy = round(accuracy_ratio * 10, 2)

    return dynamics_accuracy

def calculate_pitch_and_dynamics_accuracy(original_file, user_file, n_fft_values_pitch, n_fft_values_dynamics, pitch_tolerance=1, dynamics_tolerance_for_pitch=0.1, dynamics_tolerance=0.1, pitch_tolerance_for_dynamics=100):
    pitch_results = {}
    dynamics_results = {}
    total_pitch_accuracy = 0
    total_dynamics_accuracy = 0
    pitch_count = 0
    dynamics_count = 0

    # Calculate pitch accuracy first
    for n_fft_pitch in n_fft_values_pitch:
        original_pitches, original_dynamics = extract_pitches_and_dynamics(original_file, n_fft_pitch)
        user_pitches, user_dynamics = extract_pitches_and_dynamics(user_file, n_fft_pitch)
        
        pitch_accuracy = calculate_pitch_score(original_pitches, user_pitches, original_dynamics, user_dynamics, pitch_tolerance, dynamics_tolerance_for_pitch)
        
        pitch_results[n_fft_pitch] = {"pitch_accuracy": pitch_accuracy}
        if pitch_accuracy <= 10:  # Only include in total if not neutral
            total_pitch_accuracy += pitch_accuracy
            pitch_count += 1

    # If no values were added to total_pitch_accuracy, set average to 10 (neutral)
    average_pitch_accuracy = total_pitch_accuracy / pitch_count if pitch_count > 0 else 10

    # Calculate dynamics accuracy with dynamics n_fft values
    for n_fft_dynamics in n_fft_values_dynamics:
        original_pitches, original_dynamics = extract_pitches_and_dynamics(original_file, n_fft_dynamics)
        user_pitches, user_dynamics = extract_pitches_and_dynamics(user_file, n_fft_dynamics)
        
        dynamics_accuracy = calculate_dynamics_score(original_dynamics, user_dynamics, original_pitches, user_pitches, dynamics_tolerance, pitch_tolerance_for_dynamics)
        
        dynamics_results[n_fft_dynamics] = {"dynamics_accuracy": dynamics_accuracy}
        if dynamics_accuracy <= 10:  # Only include in total if not neutral
            total_dynamics_accuracy += dynamics_accuracy
            dynamics_count += 1

    # If no values were added to total_dynamics_accuracy, set average to 10 (neutral)
    average_dynamics_accuracy = total_dynamics_accuracy / dynamics_count if dynamics_count > 0 else 10
    
    return pitch_results, dynamics_results, average_pitch_accuracy, average_dynamics_accuracy


# Example usage
n_fft_values_pitch = [
    64, 128, 256, 384, 512, 768, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 
    2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 
    7680, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 18432, 20480, 
    22528, 24576, 26624, 28672, 30720, 32768
]

n_fft_values_dynamics = [
    64, 128, 256, 384, 512, 768, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 
    2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 
    7680, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 18432, 20480, 
    22528, 24576, 26624, 28672, 30720, 32768
]

pitch_tolerance = 0.0025 # Tolerance for pitch
dynamics_tolerance_for_pitch = 0.01 # Tolerance for dynamics when calculating pitch

dynamics_tolerance = 0.1  # Tolerance for dynamics
pitch_tolerance_for_dynamics = 0.3  # Tolerance for pitch when calculating dynamics

pitch_results, dynamics_results, average_pitch_accuracy, average_dynamics_accuracy = calculate_pitch_and_dynamics_accuracy(
    'Original_32_notes.wav', 
    '8.5.wav ', 
    n_fft_values_pitch, 
    n_fft_values_dynamics, 
    pitch_tolerance, 
    dynamics_tolerance_for_pitch,
    dynamics_tolerance,
    pitch_tolerance_for_dynamics
)

print("\nPitch Results:")
for n_fft_pitch, accuracy in pitch_results.items():
    print(f"n_fft_pitch={n_fft_pitch} -> Pitch Accuracy: {accuracy['pitch_accuracy']}/10")

print(f"\nAverage Pitch Accuracy: {average_pitch_accuracy:.2f}/10")

print("\nDynamics Results:")
for n_fft_dynamics, accuracy in dynamics_results.items():
    print(f"n_fft_dynamics={n_fft_dynamics} -> Dynamics Accuracy: {accuracy['dynamics_accuracy']}/10")
print(f"\nAverage Pitch Accuracy: {average_pitch_accuracy:.2f}/10")

print(f"\nAverage Dynamics Accuracy: {average_dynamics_accuracy:.2f}/10")
