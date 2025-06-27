from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import librosa
import io
import requests

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="tflite_learn_3.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_audio(file_stream):
    y, sr = librosa.load(file_stream, sr=16000)
    y = librosa.util.fix_length(y, size=16000)  # Adjust size to match model input
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = mfcc.T[:100]  # Example shape: (100, 40)
    mfcc = np.expand_dims(mfcc, axis=0).astype(np.float32)  # (1, 100, 40)
    return mfcc

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    audio_url = data['url']

    # Download audio file from URL
    try:
        response = requests.get(audio_url)
        response.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Failed to download audio: {str(e)}"}), 400

    # Load audio from bytes using librosa (wrap bytes in BytesIO)
    audio_bytes = io.BytesIO(response.content)
    audio = preprocess_audio(audio_bytes)

    interpreter.set_tensor(input_details[0]['index'], audio)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Optional: map output_data to label (e.g. healthy/unhealthy/uncertain)
    prediction_label = decode_output(output_data)

    return jsonify({"prediction": prediction_label})

def decode_output(output_data):
    # Implement this according to your model's output format
    # For example, if output_data is probabilities for 3 classes:
    classes = ["healthy", "unhealthy", "uncertain"]
    pred_index = np.argmax(output_data)
    return classes[pred_index]

if __name__ == "__main__":
    app.run(debug=True)
