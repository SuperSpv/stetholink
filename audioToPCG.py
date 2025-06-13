# 🧠 Install necessary libraries
!pip install pydub scipy numpy matplotlib requests

# 📚 Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
from pydub import AudioSegment

# 📥 Step 1: Get audio from user
url = input("Enter audio file URL (wav/mp3/etc): ")

# 📥 Step 2: Download the audio
filename = "downloaded_audio"
ext = url.split(".")[-1].split("?")[0]
r = requests.get(url)
with open(f"{filename}.{ext}", "wb") as f:
    f.write(r.content)

# 🔄 Step 3: Convert to WAV if not already
if ext != "wav":
    sound = AudioSegment.from_file(f"{filename}.{ext}")
    sound = sound.set_channels(1)  # mono
    sound.export(f"{filename}.wav", format="wav")
    print("Converted to WAV.")
else:
    print("Already WAV format.")

# 🎧 Step 4: Load WAV file
sample_rate, data = wavfile.read(f"{filename}.wav")

# Handle stereo by selecting one channel
if len(data.shape) == 2:
    data = data[:, 0]

# 🕒 Audio duration
duration_sec = len(data) / sample_rate
minutes = int(duration_sec // 60)
seconds = int(duration_sec % 60)
print(f"⏳ Audio duration: {duration_sec:.2f} seconds ({minutes}m {seconds}s)")

# 🔍 Step 5: Apply bandpass filter (20–200 Hz)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    return butter(order, [low, high], btype='band')

b, a = butter_bandpass(20, 200, sample_rate)
filtered = filtfilt(b, a, data)

# ⛰️ Step 6: Find peaks in filtered waveform
peaks, _ = find_peaks(filtered, distance=sample_rate*0.2, height=np.max(filtered)*0.2)

# 💓 Step 7: Detect S1 sounds only (first peak in every cardiac cycle)
min_cycle_interval = 0.6  # seconds between heartbeats
s1_peaks = []
last_beat_time = -np.inf
peak_times = peaks / sample_rate

for pt, idx in zip(peak_times, peaks):
    if pt - last_beat_time >= min_cycle_interval:
        s1_peaks.append(idx)
        last_beat_time = pt

# 📈 Step 8: Plot PCG waveform + S1 peaks with your colors
time_axis = np.arange(len(filtered)) / sample_rate

plt.figure(figsize=(15, 5))
plt.plot(time_axis, filtered, label='Filtered PCG', color='#02066f')
plt.plot(np.array(s1_peaks) / sample_rate, filtered[s1_peaks], 'o', color='#ff0000', label='Detected S1')
plt.title('Phonocardiogram with S1 Peaks')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 📊 Step 9: Estimate HR
heart_rate = len(s1_peaks) / duration_sec * 60
print(f"🫀 Estimated Heart Rate: {heart_rate:.2f} bpm (based on S1 only)")
