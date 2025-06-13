from flask import Flask, request, jsonify
import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
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
        wav_path = f"{filename}.wav"
        sound.export(wav_path, format="wav")
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

    # Plot waveform + S1 peaks
    time_axis = np.arange(len(filtered)) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, filtered, color='#02066f', label='Filtered PCG')
    plt.plot(np.array(s1_peaks) / sample_rate, filtered[s1_peaks], 'o', color='#ff0000', label='Detected S1')
    plt.title('Phonocardiogram with S1 Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot to PNG image in memory
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    img_buf.seek(0)

    # Encode image as base64 string
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

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
        "pcg_plot_png_base64": img_base64,
        "message": "Processing successful"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
