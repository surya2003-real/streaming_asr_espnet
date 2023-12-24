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


# model options
config_file = "/home/suryansh/MADHAV/asr_train_asr_raw_hindi_bpe500/exp/asr_train_asr_raw_hindi_bpe500/config.yaml"
model_file = "/home/suryansh/MADHAV/asr_train_asr_raw_hindi_bpe500/exp/asr_train_asr_raw_hindi_bpe500/valid.acc.ave_10best.pth"
device = 'cuda'
os.chdir('/home/suryansh/MADHAV/asr_train_asr_raw_hindi_bpe500')
model = Speech2Text(config_file,model_file,device=device)

dev_idx, devices = find_mics()
p = pa.PyAudio()

CHANNELS = 1
SAMPLE_RATE = 16000

#Variable initializations
buffer = []
prev_transcript = ""
transcription = ""
conf_words=[]
word_list=[]
prev_audio = []

curr_time = 1

# Callback function to read incoming data
def callback(in_data, frame_count, time_info, status):
    global buffer, transcription, conf_words, curr_time,prev_audio,initial_time
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    if curr_time==1:
        txt = transcribe(model,audio_data)
        words = txt.split()
        conf_words += words[:-4]
        buffer += words[-4:]
        curr_time += 1
        transcription += " "+" ".join(conf_words)
        print(curr_time-1, curr_time+6, transcription)
        prev_audio = audio_data
    else:
        audio_data = prev_audio + audio_data
        audio_data = audio_data[16000:]
        if(1-(time.time()-initial_time)>0):
            time.sleep(1-(time.time()-initial_time))
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
    # print(audio_data.shape)
    return (in_data, pa.paContinue)

stream = p.open(format=pa.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=dev_idx,
                    stream_callback=callback
                    )

print('Listening...')

stream.start_stream()

time.sleep(7)

initial_time = time.time()


# Keep the program running
while stream.is_active():
    try:
        pass
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
        