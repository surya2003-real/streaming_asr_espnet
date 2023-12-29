from find_microphones import find_mics
import pyaudio as pa
from ESPnet_streaming import transcribe
from espnet2.bin.asr_inference import Speech2Text
from new_conf_words import new_conf_words
import numpy as np
import os
import time
import soundfile as sf

def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

# model options
config_file = "/media/yawningwinner/New Volume/MADHAV_LAB/asr_train_asr_raw_hindi_bpe500/exp/asr_train_asr_raw_hindi_bpe500/config.yaml"
model_file = "/media/yawningwinner/New Volume/MADHAV_LAB/asr_train_asr_raw_hindi_bpe500/exp/asr_train_asr_raw_hindi_bpe500/valid.acc.ave_10best.pth"
# device = 'cuda'
os.chdir('/media/yawningwinner/New Volume/MADHAV_LAB/asr_train_asr_raw_hindi_bpe500')
model = Speech2Text(config_file,model_file)

dev_idx, devices = find_mics()
p = pa.PyAudio()

CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = 1000

#Variable initializations
buffer = []
prev_transcript = ""
transcription = ""
conf_words=[]
word_list=[]

curr_time = 1

stream = p.open(format=pa.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=dev_idx,
                    frames_per_buffer=CHUNK
                    )

print('Listening...')

stream.start_stream()

frames = []
for _ in range(0, int(SAMPLE_RATE / CHUNK * 7)):
    data = stream.read(CHUNK)
    frames.append(data)

# time.sleep(7)
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
audio_data = pcm2float(audio_data)
initial_time = time.time()
txt = transcribe(model,audio_data)
words = txt.split()
conf_words += words[:-4]
buffer += words[-4:]
curr_time += 1
transcription += " "+" ".join(conf_words)
print(curr_time-1, curr_time+6, transcription)

# Keep the program running
while True:
    try:
        new_frames = []
        # if(1-(time.time()-initial_time)>0):
        #     time.sleep(1-(time.time()-initial_time))
        for _ in range(0, int(SAMPLE_RATE / CHUNK * 1)):
            data = stream.read(CHUNK)
            frames.append(data)
            new_frames.append(data)
        new_data = np.frombuffer(b''.join(new_frames), dtype=np.int16)
        new_data = pcm2float(new_data)
        audio_data=np.append(audio_data,new_data)
        print("audio_data: ", audio_data)
        transcription_data = audio_data[-int(SAMPLE_RATE/CHUNK)*7:]
        start_time = time.time()
        # if a is None:
        #     print("break here")
        #     break
        txt = transcribe(model,transcription_data)
        print(txt)
        curr_time += 1
        word_list = txt.split()
        print(word_list)
        word_list=word_list[:-1]
        conf_words,buffer,temp = new_conf_words(buffer,word_list,conf_words)
        if(len(temp)>0):
            transcription += " "+" ".join(temp)
        print(curr_time-1, curr_time+6, transcription, time.time()-start_time)
        prev_audio = audio_data
        initial_time = time.time()
    except KeyboardInterrupt:
        break

# Stop stream
stream.stop_stream()
stream.close()

# Terminate PyAudio
p.terminate()
print("stream stopped")

transcription += " "+" ".join(buffer)
print(transcription)
print("ye rhaaa")

filename = "outpu.wav"
print(audio_data)

        # Save the audio data to a WAV file
sf.write(filename, audio_data,SAMPLE_RATE)

print(f"Audio data has been successfully saved to {filename}")
