import os
import numpy as np
import librosa
import requests
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="tflite_learn_3.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

TARGET_SAMPLE_RATE = 16000
TARGET_DURATION = 1.5  # seconds
TARGET_LENGTH = int(TARGET_SAMPLE_RATE * TARGET_DURATION)  # 24000 samples

def download_audio(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download audio: {response.status_code}")
    filename = "temp.wav"
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
    if len(audio) < TARGET_LENGTH:
        audio = librosa.util.fix_length(audio, size=TARGET_LENGTH)
    else:
        audio = audio[:TARGET_LENGTH]
    return np.array(audio, dtype=np.float32).reshape(1, TARGET_LENGTH)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request"}), 400

    try:
        file_path = download_audio(data["url"])
        input_data = preprocess_audio(file_path)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        labels = ["healthy", "uncertain", "unhealthy"]
        result = {labels[i]: round(output[0][i], 4) for i in range(len(labels))}

        os.remove(file_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
