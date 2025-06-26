from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

EDGE_API = "https://ingestion.edgeimpulse.com/api/classify"
API_KEY = "ei_95b7e4358fe3394447ab884554410c1b5c1c77ab8debab6a6c78e7e5a522c0bf"

@app.route("/classify", methods=["POST"])
def classify():
    print("Raw data:", request.data)
    try:
        data = request.get_json(force=True)
        print("JSON data:", data)
    except Exception as e:
        return jsonify({"error": "Invalid JSON: " + str(e)}), 400

    audio_url = data.get("audio_url")
    if not audio_url:
        return jsonify({"error": "Missing 'audio_url'"}), 400

    try:
        audio_data = requests.get(audio_url).content
        response = requests.post(
            EDGE_API,
            headers={
                "x-api-key": API_KEY,
                "Content-Type": "audio/wav"
            },
            data=audio_data
        )
        return jsonify(response.json())
    except Exception as e:
        print("Error occurred:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
