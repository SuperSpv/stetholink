from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import librosa
import io

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
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    audio = preprocess_audio(file)

    interpreter.set_tensor(input_details[0]['index'], audio)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return jsonify({"prediction": output_data.tolist()})
