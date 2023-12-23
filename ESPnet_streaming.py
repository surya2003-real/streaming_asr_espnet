import librosa  
from functools import lru_cache
import time
from espnet2.bin.asr_inference import Speech2Text
from new_conf_words import new_conf_words
import os
import pandas as pd
import torch
from IPython.display import Audio
from pprint import pprint
import numpy as np
# Function to load audio file
@lru_cache
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000)
    return a

# Function to load an audio chunk from the larger file
def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]


def initial_audio(model,audio):
    global buffer
    global transcription
    text = transcribe(model,audio)
    return text

# Function to transcribe audio chunk
def transcribe(model,audio):
    nbest = model(audio)
    return nbest[0][0]

def generate_transcription(audio_path,config_file,model_file,device='cuda'):
    # Modify the following line based on the path to the config file and model file
    os.chdir('/home/suryansh/MADHAV/asr_train_asr_raw_hindi_bpe500')
    speech2text = Speech2Text(config_file,model_file,device=device)
    torch.set_num_threads(1)
    # download example
    torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=True)
    (get_speech_timestamps,
    _, read_audio,
    *_) = utils
    sampling_rate=16000
    #Variable initializations
    buffer = []
    transcription = ""
    conf_words=[]
    word_list=[]
    SAMPLING_RATE = 16000
    audio_len = len(load_audio(audio_path))
    duration = audio_len/SAMPLING_RATE
    print("Audio duration is: %2.2f seconds" % duration)

    initial_time = time.time()
    curr_time=0
    second_count = 0
    # for i in range(0,0):
    #     start_time = time.time()
    #     a = load_audio_chunk(audio_path,i,i+1)
    #     txt = initial_audio(speech2text,a)
    #     transcription += " "+txt
    #     conf_words.append(txt)
    #     print(transcription, time.time()-start_time)    
    #     second_count += 1
    #     delay = second_count - (time.time() - initial_time)
    #     if delay > 0:
    #         time.sleep(delay)
    #     print(second_count)

    if duration<8:
        a = load_audio_chunk(audio_path,0,duration)
        txt = initial_audio(speech2text,a)
        transcription += " "+txt
        return transcription
    
    # The first 7 seconds of the audio file are transcribed
    a = load_audio_chunk(audio_path,curr_time,min(curr_time+7, duration))
    txt = initial_audio(speech2text,a)
    words = txt.split()
    conf_words += words[:-4]
    buffer += words[-4:]
    curr_time += 1
    transcription += " "+" ".join(conf_words)
    # curr_time = 1
    print(curr_time-1, min(curr_time+6, duration), transcription)
    print("breakdown",conf_words, buffer)
    speech = np.array([])
    while curr_time+7<duration:
        start_time = time.time()
        a = load_audio_chunk(audio_path,curr_time,min(curr_time+7, duration))
        speech_timestamps = get_speech_timestamps(a, model, sampling_rate=sampling_rate, threshold=0.2)
        df = pd.DataFrame(speech_timestamps)
        df = df // 16
        print(df)
        speech = np.array([])
        duration=7*1000
        for _, row in df.iterrows():
            start_sample = row['start']
            end_sample = row['end']
    #         print(start_sample, end_sample, duration)
            if(start_sample<0 and end_sample<0):
                continue
            if(start_sample>duration and end_sample>duration):
                break
    #         print("Y")
            speech = np.concatenate([speech, a[int(max(0,start_sample))*16:int(min(duration, end_sample))*16]])
        print(len(speech))
        if(len(speech)>1000):
            txt = initial_audio(speech2text,speech)
            print(txt)
            curr_time += 1
            word_list = txt.split()
            print(word_list)
            word_list=word_list[:-1]
            conf_words,buffer,temp = new_conf_words(buffer,word_list,conf_words)
            if(len(temp)>0):
                transcription += " "+" ".join(temp)
            print(curr_time-1, min(curr_time+6, duration), transcription, time.time()-start_time)
            print("breakdown",conf_words, buffer)
        second_count+=1
        delay = second_count - (time.time() - initial_time)
        if delay > 0:
            time.sleep(delay)
        print(second_count)
    transcription += " "+" ".join(buffer)
    return transcription
