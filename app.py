from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
from pydub import AudioSegment
import base64
import io
import os

app = FastAPI()


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    return butter(order, [low, high], btype='band')


@app.get("/analyze")
def analyze_audio(url: str = Query(..., description="URL to audio file (mp3, wav, etc.)")):
    filename = "downloaded_audio"
    ext = url.split(".")[-1].split("?")[0]

    # Download audio
    try:
        r = requests.get(url)
        with open(f"{filename}.{ext}", "wb") as f:
            f.write(r.content)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to download file: {e}"})

    # Convert to WAV if needed
    wav_path = f"{filename}.wav"
    if ext != "wav":
        sound = AudioSegment.from_file(f"{filename}.{ext}")
        sound = sound.set_channels(1)
        sound.export(wav_path, format="wav")
        os.remove(f"{filename}.{ext}")
    else:
        os.rename(f"{filename}.{ext}", wav_path)

    # Read WAV
    sample_rate, data = wavfile.read(wav_path)
    if len(data.shape) == 2:
        data = data[:, 0]

    duration_sec = len(data) / sample_rate

    # Filter
    b, a = butter_bandpass(20, 200, sample_rate)
    filtered = filtfilt(b, a, data)

    # Peaks (detecting S1)
    peaks, _ = find_peaks(filtered, distance=sample_rate * 0.2, height=np.max(filtered) * 0.2)
    s1_peaks = []
    last_beat_time = -np.inf
    for pt, idx in zip(peaks / sample_rate, peaks):
        if pt - last_beat_time >= 0.6:
            s1_peaks.append(idx)
            last_beat_time = pt

    # Heart Rate Calculation
    heart_rate = len(s1_peaks) / duration_sec * 60

    # Generate PCG plot
    time_axis = np.arange(len(filtered)) / sample_rate
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time_axis, filtered, label="Filtered PCG", color="#02066f")
    ax.plot(np.array(s1_peaks) / sample_rate, filtered[s1_peaks], 'o', color="#ff0000", label="Detected S1")
    ax.set_title("Phonocardiogram with S1 Peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode("utf-8")

    # Clean up
    os.remove(wav_path)

    return {
        "audio_duration_sec": round(duration_sec, 2),
        "heart_rate_bpm": round(heart_rate, 2),
        "pcg_plot_png_base64": img_base64
    }
