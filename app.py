from flask import Flask, request, jsonify
import requests
import librosa
import numpy as np
import tensorflow as tf
import tempfile
import os

app = Flask(__name__)

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="tflite_learn_3.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ['healthy', 'unhealthy', 'uncertain']  # adjust if needed

@app.route("/diagnose", methods=["POST"])
def diagnose():
    try:
        # Get audio URL from request
        audio_url = request.json.get("audio_url")
        if not audio_url:
            return jsonify({"error": "Missing audio_url"}), 400

        # Download audio from the URL
        response = requests.get(audio_url)
        if response.status_code != 200:
            return jsonify({"error": "Audio download failed"}), 400

        # Save to a temporary .wav file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        # Load audio using librosa and resample to 16kHz mono
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        os.remove(tmp_path)  # cleanup

        # Split into 1-second chunks
        num_segments = len(y) // 16000
        counts = {label: 0 for label in labels}

        for i in range(num_segments):
            segment = y[i*16000 : (i+1)*16000]

            # Extract MFCCs (13 coefficients, 75 frames = 975 features)
            mfcc = librosa.feature.mfcc(
                y=segment,
                sr=16000,
                n_mfcc=13,
                n_fft=400,     # 25ms window
                hop_length=213 # ~75 frames per second
            )

            features = mfcc.flatten().astype(np.float32).reshape(1, -1)

            # Validate shape matches model's input
            expected_shape = input_details[0]['shape'][1]
            if features.shape[1] != expected_shape:
                return jsonify({
                    "error": f"MFCC shape mismatch: got {features.shape[1]}, expected {expected_shape}"
                }), 400

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            class_index = int(np.argmax(output[0]))
            counts[labels[class_index]] += 1

        return jsonify(counts)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
