from flask import Flask, request, jsonify
import requests
import librosa
import numpy as np
import tensorflow as tf
import tempfile
import os
import logging

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# --- Model Loading ---
try:
    # Load your TFLite model
    interpreter = tf.lite.Interpreter(model_path="tflite_learn_3.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("✅ TFLite model loaded successfully.")
    # Getting the expected input shape from the model itself
    EXPECTED_INPUT_SHAPE = input_details[0]['shape']
    logging.info(f"Model expects input shape: {EXPECTED_INPUT_SHAPE}")
except Exception as e:
    logging.error(f"❌ CRITICAL: Failed to load TFLite model. Error: {e}")
    # If the model doesn't load, the app can't work.
    interpreter = None

# Labels from your Edge Impulse project
labels = ['healthy', 'unhealthy'] 

# --- Parameters MATCHING Edge Impulse ---
SAMPLE_RATE = 4000   # CRITICAL: Changed to 4000 Hz to match your training
WINDOW_SIZE_S = 1.5  # 1500 ms
STRIDE_S = 0.5       # 500 ms

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SIZE_S)
STRIDE_SAMPLES = int(SAMPLE_RATE * STRIDE_S)

@app.route("/", methods=["GET"])
def home():
    return "✅ StethoLink API is live and model is loaded."

@app.route("/diagnose", methods=["POST"])
def diagnose():
    if interpreter is None:
        logging.error("Diagnosis attempted but model is not loaded.")
        return jsonify({"error": "Model not loaded, check server logs"}), 500
        
    try:
        json_data = request.get_json(force=True, silent=True)
        if not json_data:
            return jsonify({"error": "Request body is not valid JSON"}), 400

        audio_url = json_data.get("audio_url")
        if not audio_url:
            return jsonify({"error": "Missing 'audio_url'"}), 400

        logging.info(f"Received request for URL: {audio_url}")
        
        # Download audio
        response = requests.get(audio_url)
        if response.status_code != 200:
            logging.error(f"Audio download failed with status code: {response.status_code}")
            return jsonify({"error": "Audio download failed"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        # Load audio with the CORRECT sample rate
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        os.remove(tmp_path)
        logging.info(f"Audio loaded. Duration: {len(y)/SAMPLE_RATE:.2f}s")

        if len(y) < WINDOW_SAMPLES:
            logging.warning("Audio clip is shorter than 1.5s, cannot process.")
            return jsonify({"error": "Audio clip is too short. Must be at least 1.5 seconds."}), 400

        counts = {label: 0 for label in labels}
        
        # --- Sliding Window Logic ---
        num_windows = 0
        for i in range(0, len(y) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            num_windows += 1
            segment = y[i : i + WINDOW_SAMPLES]

            # Extract features for the 1.5-second window
            # These MFCC parameters are standard; adjust if your project used custom ones.
            mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=13)
            
            features = mfcc.flatten().astype(np.float32)

            # Reshape for the model, assuming the model wants a flat vector.
            # E.g., (1, 975) for a 1.5s window at 4kHz with these MFCC settings.
            # We use the shape the model told us it expects.
            features = features.reshape(EXPECTED_INPUT_SHAPE)

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            class_index = int(np.argmax(output[0]))
            
            if class_index < len(labels):
                prediction = labels[class_index]
                counts[prediction] += 1
            else:
                logging.warning(f"Model predicted an out-of-bounds index: {class_index}")
        
        logging.info(f"Processed {num_windows} windows. Counts: {counts}")
        return jsonify(counts)

    except Exception as e:
        logging.exception("❌ Error during diagnosis")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import sys
    if "gunicorn" not in sys.modules:
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)
