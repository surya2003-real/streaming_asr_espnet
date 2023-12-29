import sounddevice as sd
import wave

def record_audio(filename, duration=5, samplerate=16000):
    print("Recording...")

    # Record audio from the default microphone
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()

    print("Recording complete.")

    # Save the audio data to a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Stereo audio
        wf.setsampwidth(2)  # 16-bit sample width
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    print(f"Audio saved to {filename}")

if __name__ == "__main__":
    output_filename = "recorded_audio.wav"
    record_duration = 5  # in seconds

    record_audio(output_filename, duration=record_duration)
