from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import os

app = Flask(__name__, template_folder='.')

def calculate_pitch_accuracy(original_file, user_file):
    def detect_pitch_yin(audio_file):
        y, sr = librosa.load(audio_file)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = [pitch for pitch in pitches[magnitudes > 0] if pitch > 0]
        return pitches
    
    def calculate_pitch_score(original_pitches, user_pitches):
        threshold = 0.3
        correct_notes = 0
        
        for orig_pitch in original_pitches:
            if any(abs(orig_pitch - user_pitch) < threshold for user_pitch in user_pitches):
                correct_notes += 1
        
        total_notes = len(original_pitches)
        if total_notes == 0:
            return 0.0
        
        accuracy_ratio = correct_notes / total_notes
        pitch_accuracy = round((accuracy_ratio ** 2) * 10, 1)
        
        return pitch_accuracy
    
    original_pitches = detect_pitch_yin(original_file)
    user_pitches = detect_pitch_yin(user_file)
    
    if not original_pitches:
        return {"pitch_accuracy": 0.0}
    
    pitch_accuracy = calculate_pitch_score(original_pitches, user_pitches)
    
    return {"pitch_accuracy": pitch_accuracy}

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
        tempo = np.mean(original_diffs) if len(original_diffs) > 0 else 1
        
        threshold = tempo * 0.2
        
        correct_timing = 0
        
        for orig_onset in original_onsets:
            if any(abs(orig_onset - user_onset) <= threshold for user_onset in user_onsets):
                correct_timing += 1
        
        total_onsets = len(original_onsets)
        if total_onsets == 0:
            return 0.0
        
        accuracy_ratio = correct_timing / total_onsets
        timing_accuracy = round(accuracy_ratio * 11, 1)
        timing_accuracy = min(timing_accuracy, 10.0)
        
        return timing_accuracy
    
    original_onsets = detect_onsets(original_file)
    user_onsets = detect_onsets(user_file)
    
    timing_accuracy = calculate_timing_score(np.array(original_onsets), np.array(user_onsets))
    
    return {"timing_accuracy": timing_accuracy}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        original_file = request.files['original_file']
        user_file = request.files['user_file']
        
        original_path = "temp_original.wav"
        user_path = "temp_user.wav"
        
        original_file.save(original_path)
        user_file.save(user_path)
        
        pitch_result = calculate_pitch_accuracy(original_path, user_path)
        timing_result = calculate_timing_accuracy(original_path, user_path)
        
        result = {
            'pitch_accuracy': pitch_result['pitch_accuracy'],
            'timing_accuracy': timing_result['timing_accuracy']
        }
        
        # Clean up temporary files
        os.remove(original_path)
        os.remove(user_path)

        return jsonify(result)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
