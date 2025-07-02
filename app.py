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

labels = ['healthy', 'unhealthy', 'uncertain']  # Adjust if needed

@app.route("/diagnose", methods=["POST"])
def diagnose():
    try:
        logging.info(f"Headers: {dict(request.headers)}")
        logging.info(f"Raw data: {request.data}")

        # Use get_json with force=True to parse even if Content-Type is missing
        json_data = request.get_json(force=True, silent=True)
        logging.info(f"Parsed JSON: {json_data}")

        if not json_data:
            return jsonify({"error": "Request body is not valid JSON"}), 400

        audio_url = json_data.get("audio_url")
        if not audio_url:
            return jsonify({"error": "Missing or invalid 'audio_url' in JSON body"}), 400

        # Download audio file from URL
        response = requests.get(audio_url)
        if response.status_code != 200:
            return jsonify({"error": "Audio download failed"}), 400

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        # Load and resample audio to 16kHz mono
        y, sr = librosa.load(tmp_path, sr=16000, mono=True)
        os.remove(tmp_path)  # clean up temp file

        # Limit audio to max 15 seconds to avoid timeout
        max_duration_sec = 15
        max_samples = max_duration_sec * 16000
        if len(y) > max_samples:
            y = y[:max_samples]

        logging.info(f"Audio length (seconds): {len(y)/16000:.2f}")

        # Split audio into 1-second segments
        num_segments = len(y) // 16000
        counts = {label: 0 for label in labels}

        for i in range(num_segments):
            segment = y[i*16000 : (i+1)*16000]

            # Extract MFCC features (13 coeffs, 75 frames = 975 features)
            mfcc = librosa.feature.mfcc(
                y=segment,
                sr=16000,
                n_mfcc=13,
                n_fft=400,
                hop_length=213
            )
            features = mfcc.flatten().astype(np.float32).reshape(1, -1)

            expected_shape = input_details[0]['shape'][1]
            if features.shape[1] != expected_shape:
                return jsonify({
                    "error": f"MFCC shape mismatch: got {features.shape[1]}, expected {expected_shape}"
                }), 400

            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            class_index = int(np.argmax(output[0]))
            counts[labels[class_index]] += 1

        return jsonify(counts)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
