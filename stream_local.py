from find_microphones import find_mics
import pyaudio as pa
from ESPnet_streaming import transcribe
from espnet2.bin.asr_inference import Speech2Text
from new_conf_words import new_conf_words
import librosa
import numpy as np
import os
import sys
import time
from scipy.io.wavfile import write
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
CHUNK = 1024

#Variable initializations
buffer = []
prev_transcript = ""
transcription = ""
conf_words=[]
word_list=[]

curr_time = 1

# Callback function to read incoming data
# def callback(in_data, frame_count, time_info, status):
#     global buffer, transcription, conf_words, curr_time,prev_audio,initial_time
#     audio_data = np.frombuffer(in_data, dtype=np.int16)
#     if curr_time==1:
#         txt = transcribe(model,audio_data)
#         words = txt.split()
#         conf_words += words[:-4]
#         buffer += words[-4:]
#         curr_time += 1
#         transcription += " "+" ".join(conf_words)
#         print(curr_time-1, curr_time+6, transcription)
#         prev_audio = audio_data
#     else:
#         audio_data = prev_audio + audio_data
#         audio_data = audio_data[16000:]
#         # if(1-(time.time()-initial_time)>0):
#         #     time.sleep(1-(time.time()-initial_time))
#         start_time = time.time()
#         # if a is None:
#         #     print("break here")
#         #     break
#         txt = transcribe(model,audio_data)
#         print(txt)
#         curr_time += 1
#         word_list = txt.split()
#         print(word_list)
#         word_list=word_list[:-1]
#         conf_words,buffer,temp = new_conf_words(buffer,word_list,conf_words)
#         if(len(temp)>0):
#             transcription += " "+" ".join(temp)
#         print(curr_time-1, curr_time+6, transcription, time.time()-start_time)
#         prev_audio = audio_data
#         initial_time = time.time()
#     # print(audio_data.shape)
#     return (in_data, pa.paContinue)

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
        if(1-(time.time()-initial_time)>0):
            time.sleep(1-(time.time()-initial_time))
        for _ in range(0, int(SAMPLE_RATE / CHUNK * 1)):
            data = stream.read(CHUNK)
            frames.append(data)
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        print("audio_data: ", audio_data)
        audio_data = audio_data[-16000*7:]
        start_time = time.time()
        # if a is None:
        #     print("break here")
        #     break
        txt = transcribe(model,audio_data)
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
sample_rate = 16000  # Example sample rate (Hz)
        # audio_data = np.random.uniform(-1, 1, size=(sample_rate * 5,))  # 5 seconds of random audio

        # Specify the file name
filename = "outpu.wav"
print(audio_data)

        # Save the audio data to a WAV file
write(filename, sample_rate, audio_data.astype(np.int16))

print(f"Audio data has been successfully saved to {filename}")
        


# try:
#     while stream.is_active():
#         if(1-(time.time()-initial_time)>0):
#             time.sleep(1-(time.time()-initial_time))
#         start_time = time.time()
#         a = receive_audio_chunk()
#         running_audio = a[curr_time*16000:]
#         # if a is None:
#         #     print("break here")
#         #     break
#         txt = transcribe(model,running_audio)
#         print(txt)
#         curr_time += 1
#         word_list = txt.split()
#         print(word_list)
#         word_list=word_list[:-1]
#         conf_words,buffer,temp = new_conf_words(buffer,word_list,conf_words)
#         if(len(temp)>0):
#             transcription += " "+" ".join(temp)
#         print(curr_time-1, curr_time+6, transcription, time.time()-start_time)
#         initial_time = time.time()
# except KeyboardInterrupt:
#     stream.stop_stream()
#     print("stream stopped")
# transcription += " "+" ".join(buffer)
# print(transcription)
        