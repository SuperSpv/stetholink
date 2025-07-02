from flask import Flask, request, jsonify
import requests
import librosa
import numpy as np
import tensorflow as tf
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load your TFLite model
interpreter = tf.lite.Interpreter(model_path="tflite_learn_3.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels from your Edge Impulse project. Add 'uncertain' if you have it.
labels = ['healthy', 'unhealthy'] 

# --- Parameters from Edge Impulse ---
SAMPLE_RATE = 16000  # We'll use 16k Hz as it's common, though your project shows 4k. Adjust if needed.
WINDOW_SIZE_S = 1.5  # 1500 ms
STRIDE_S = 0.5       # 500 ms

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SIZE_S)
STRIDE_SAMPLES = int(SAMPLE_RATE * STRIDE_S)

@app.route("/", methods=["GET"])
def home():
    return "✅ StethoLink API is live"

@app.route("/diagnose", methods=["POST"])
def diagnose():
    try:
        json_data = request.get_json(force=True, silent=True)
        if not json_data:
            return jsonify({"error": "Request body is not valid JSON"}), 400

        audio_url = json_data.get("audio_url")
        if not audio_url:
            return jsonify({"error": "Missing 'audio_url'"}), 400

        # Download audio
        response = requests.get(audio_url)
        if response.status_code != 200:
            return jsonify({"error": "Audio download failed"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        # Load audio
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        os.remove(tmp_path)

        if len(y) < WINDOW_SAMPLES:
            return jsonify({"error": "Audio clip is too short. Must be at least 1.5 seconds."}), 400

        # Dictionary to count predictions
        counts = {label: 0 for label in labels}
        
        # --- Sliding Window Logic ---
        for i in range(0, len(y) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            segment = y[i : i + WINDOW_SAMPLES]

            # Extract MFCCs for the 1.5-second window
            mfcc = librosa.feature.mfcc(
                y=segment,
                sr=SAMPLE_RATE,
                n_mfcc=13,
                n_fft=400,
                hop_length=213
            )
            # This shape adjustment might be needed depending on the exact output of Edge Impulse
            mfcc = mfcc[:, :75] 

            features = mfcc.flatten().astype(np.float32).reshape(1, -1)

            # Ensure the feature shape matches the model's expected input
            expected_shape = input_details[0]['shape'][1]
            if features.shape[1] != expected_shape:
                 return jsonify({"error": f"MFCC shape mismatch: got {features.shape[1]}, expected {expected_shape}"}), 500

            # Run inference on the window
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            class_index = int(np.argmax(output[0]))
            
            # Ensure class_index is within the bounds of your labels list
            if class_index < len(labels):
                prediction = labels[class_index]
                counts[prediction] += 1
            else:
                logging.warning(f"Model predicted an out-of-bounds index: {class_index}")

        # Return the total counts from all windows
        return jsonify(counts)

    except Exception as e:
        logging.exception("Error during diagnosis")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import sys
    if "gunicorn" not in sys.modules:
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)
