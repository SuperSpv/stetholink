from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# رابط API من Edge Impulse
EDGE_API = "https://ingestion.edgeimpulse.com/api/classify"

# مفتاح API حقك
API_KEY = "ei_95b7e4358fe3394447ab884554410c1b5c1c77ab8debab6a6c78e7e5a522c0bf"

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    audio_url = data.get("audio_url")

    if not audio_url:
        return jsonify({"error": "Missing audio_url"}), 400

    try:
        # حمل ملف الصوت من الرابط
        audio_data = requests.get(audio_url).content

        # أرسل ملف الصوت إلى Edge Impulse
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
        return jsonify({"error": str(e)}), 500

# هذا الجزء يخلي السيرفر يشتغل صح على Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
