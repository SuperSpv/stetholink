from flask import Flask, request, jsonify
import requests
import os
import traceback

app = Flask(__name__)

EDGE_API = "https://ingestion.edgeimpulse.com/api/classify"
API_KEY = "ei_95b7e4358fe3394447ab884554410c1b5c1c77ab8debab6a6c78e7e5a522c0bf"

@app.route("/classify", methods=["POST"])
def classify():
    try:
        # نتوقع ملف صوت تحت اسم 'file' في الفورم
        if 'file' not in request.files:
            return jsonify({"error": "Missing file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        audio_data = file.read()

        # نرسل الملف مباشرة لـ Edge Impulse
        response = requests.post(
            EDGE_API,
            headers={
                "x-api-key": API_KEY,
                "Content-Type": "audio/wav"
            },
            data=audio_data
        )

        if response.status_code != 200:
            return jsonify({
                "error": "Edge Impulse API error",
                "status_code": response.status_code,
                "response_text": response.text
            }), 500

        return jsonify(response.json())

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
