from flask import Flask, request, jsonify
import requests
import numpy as np
import tensorflow as tf
import tempfile
import os
import librosa
import logging

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Load TFLite model (with DSP included)
interpreter = tf.lite.Interpreter(model_path="tflite_learn_3.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ['healthy', 'unhealthy', 'uncertain']  # Adjust if needed

@app.route("/diagnose", methods=["POST"])
def diagnose():
    try:
        data = request.get_json(force=True)
        audio_url = data.get("audio_url")
        if not audio_url:
            return jsonify({"error": "Missing 'audio_url'"}), 400

        # Download audio
        response = requests.get(audio_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download audio"}), 400

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load audio as mono, 16kHz
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
        os.remove(tmp_path)

        # Segment size: 1.5 sec = 24000 samples
        segment_samples = 24000
        total_segments = len(audio) // segment_samples
        counts = {label: 0 for label in labels}

        for i in range(total_segments):
            segment = audio[i*segment_samples:(i+1)*segment_samples]
            segment = segment.astype(np.float32).reshape(1, -1)

            expected_input_shape = input_details[0]['shape'][1]
            if segment.shape[1] != expected_input_shape:
                return jsonify({"error": f"Shape mismatch: got {segment.shape[1]}, expected {expected_input_shape}"}), 400

            interpreter.set_tensor(input_details[0]['index'], segment)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            predicted_index = int(np.argmax(output[0]))
            counts[labels[predicted_index]] += 1

        return jsonify(counts)

    except Exception as e:
        logging.exception("Unexpected error")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
