import numpy as np
from scipy.io.wavfile import write

# Sample audio data (replace this with your actual NumPy array)
sample_rate = 16000  # Example sample rate (Hz)
audio_data = np.random.uniform(-100, 100, size=(sample_rate * 5,))  # 5 seconds of random audio

# Specify the file name
filename = "output.wav"

# Save the audio data to a WAV file
write(filename, sample_rate, audio_data.astype(np.int16))

print(f"Audio data has been successfully saved to {filename}")
