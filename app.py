import os
import numpy as np
import librosa
import requests
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="tflite_learn_3.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Constants
SR = 16000
DURATION = 1.5           # seconds
SAMPLES = int(SR * DURATION)  # 24000
N_MFCC = 13
N_FRAMES = 75
FEATURE_LEN = N_MFCC * N_FRAMES  # 975

def download_audio(url):
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"Failed to download audio: {r.status_code}")
    path = "temp_audio.wav"
    with open(path, "wb") as f:
        f.write(r.content)
    return path

def extract_features(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    if len(y) < SAMPLES:
        y = librosa.util.fix_length(y, size=SAMPLES)
    else:
        y = y[:SAMPLES]
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=400, hop_length=int(SAMPLES / (N_FRAMES - 1)))
    mfcc = mfcc[:, :N_FRAMES]
    feat = mfcc.flatten()
    if feat.size < FEATURE_LEN:
        feat = np.pad(feat, (0, FEATURE_LEN - feat.size))
    else:
        feat = feat[:FEATURE_LEN]
    return np.expand_dims(feat.astype(np.float32), axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data or "url" not in data:
        return jsonify(error="Missing 'url'"), 400
    try:
        filepath = download_audio(data["url"])
        features = extract_features(filepath)
        os.remove(filepath)
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        labels = ["healthy", "unhealthy", "uncertain"]
        result = {labels[i]: round(float(output[i]), 4) for i in range(len(output))}
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
