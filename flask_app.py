from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="tflite_learn_3.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("input")
    if not data:
        return jsonify({"error": "No input provided"}), 400

    # Convert input to numpy array and reshape
    input_data = np.array(data, dtype=np.float32)
    input_data = input_data.reshape(input_details[0]["shape"])

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return jsonify({"prediction": output.tolist()})
