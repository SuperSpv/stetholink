from flask import Flask, request, jsonify
import numpy as np
import requests
from io import BytesIO
from pydub import AudioSegment
import tensorflow as tf
import librosa

app = Flask(__name__)

# Load and initialize the TFLite model once at startup
MODEL_PATH = "tflite_learn_3.tflite"  # Path to your TFLite model file
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()  # Allocate input & output tensors:contentReference[oaicite:6]{index=6}
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def download_audio(url):
    """Download audio from URL and return as a pydub AudioSegment."""
    resp = requests.get(url)
    resp.raise_for_status()
    audio = AudioSegment.from_file(BytesIO(resp.content))  # auto-detect format (requires ffmpeg)
    return audio

def preprocess_audio(audio):
    """
    Convert to mono 4000 Hz and return numpy array of samples (float32, range [-1,1]).
    Edge Impulse impulse uses 4000 Hz sampling (as per design). 
    """
    audio = audio.set_channels(1)            # mono
    audio = audio.set_frame_rate(4000)       # 4000 Hz:contentReference[oaicite:7]{index=7}
    samples = np.array(audio.get_array_of_samples())
    # Convert from int16 (pydub) to float32 in [-1,1]
    if audio.sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    else:
        samples = samples.astype(np.float32)
    return samples

def extract_mfcc(samples):
    """
    Compute MFCC features (13 coeffs). Returns a 1D array of length 975 (13*75).
    Uses librosa with n_mfcc=13. Pads/trims to 975 if needed.
    """
    # Compute MFCC with librosa: output shape = (n_mfcc, t):contentReference[oaicite:8]{index=8}
    mfcc = librosa.feature.mfcc(y=samples, sr=4000, n_mfcc=13, n_fft=80, hop_length=80)
    # Flatten in column-major order (frames as columns)
    feat = mfcc.flatten(order='F')  # yields length n_mfcc * n_frames
    # Ensure length 975 (pad or trim)
    if feat.size < 975:
        feat = np.pad(feat, (0, 975 - feat.size), mode='constant')
    else:
        feat = feat[:975]
    return feat.astype(np.float32)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify(error="Missing 'url' in request"), 400

    url = data['url']
    try:
        audio = download_audio(url)
    except Exception as e:
        return jsonify(error=f"Failed to download or decode audio: {e}"), 400

    samples = preprocess_audio(audio)
    duration_ms = len(audio)  # total duration in ms
    window_ms = 1500
    step_ms = 500

    healthy_count = 0
    unhealthy_count = 0
    uncertain_count = 0

    # Slide over the audio in 0.5s steps, 1.5s window
    for start_ms in range(0, duration_ms, step_ms):
        end_ms = start_ms + window_ms
        segment = audio[start_ms:end_ms]
        # If shorter than 1.5s, pad with silence (zero)
        if len(segment) < window_ms:
            silence = AudioSegment.silent(duration=(window_ms - len(segment)))
            segment = segment + silence
        seg_samples = preprocess_audio(segment)
        # Compute MFCC features
        features = extract_mfcc(seg_samples)
        # Prepare input tensor (shape should match model: e.g., [1, 975])
        input_data = np.array(features, dtype=np.float32).reshape(1, -1)
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        # Assume output_data is [healthy_prob, unhealthy_prob]
        healthy_prob, unhealthy_prob = output_data[0], output_data[1]

        # Simple thresholding for "uncertain"
        if max(healthy_prob, unhealthy_prob) < 0.6:
            uncertain_count += 1
        elif healthy_prob >= unhealthy_prob:
            healthy_count += 1
        else:
            unhealthy_count += 1

    result = {"healthy": healthy_count, "unhealthy": unhealthy_count, "uncertain": uncertain_count}
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
