from flask import Flask, request, jsonify
import os
import numpy as np
import requests
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
from pydub import AudioSegment

app = Flask(__name__)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    return butter(order, [low, high], btype='band')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    data = request.json
    url = data.get('audio_url')

    if not url:
        return jsonify({"error": "Missing audio_url"}), 400

    filename = "downloaded_audio"
    ext = url.split(".")[-1].split("?")[0]

    # Download audio
    r = requests.get(url)
    if r.status_code != 200:
        return jsonify({"error": "Failed to download audio"}), 400
    with open(f"{filename}.{ext}", "wb") as f:
        f.write(r.content)

    # Convert to wav if needed
    if ext != "wav":
        sound = AudioSegment.from_file(f"{filename}.{ext}")
        sound = sound.set_channels(1)  # mono
        sound.export(f"{filename}.wav", format="wav")
        wav_path = f"{filename}.wav"
    else:
        wav_path = f"{filename}.wav"

    # Read WAV
    sample_rate, data = wavfile.read(wav_path)
    if len(data.shape) == 2:
        data = data[:, 0]

    # Duration
    duration_sec = len(data) / sample_rate

    # Filter
    b, a = butter_bandpass(20, 200, sample_rate)
    filtered = filtfilt(b, a, data)

    # Peaks
    peaks, _ = find_peaks(filtered, distance=sample_rate*0.2, height=np.max(filtered)*0.2)

    # S1 detection
    min_cycle_interval = 0.6
    s1_peaks = []
    last_beat_time = -np.inf
    peak_times = peaks / sample_rate

    for pt, idx in zip(peak_times, peaks):
        if pt - last_beat_time >= min_cycle_interval:
            s1_peaks.append(idx)
            last_beat_time = pt

    # Heart rate
    heart_rate = len(s1_peaks) / duration_sec * 60

    # Clean up files
    try:
        os.remove(f"{filename}.{ext}")
        os.remove(wav_path)
    except Exception:
        pass

    return jsonify({
        "heart_rate": round(heart_rate, 2),
        "duration_seconds": round(duration_sec, 2),
        "s1_peaks_count": len(s1_peaks),
        "message": "Processing successful"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
