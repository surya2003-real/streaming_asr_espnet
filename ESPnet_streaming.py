import sys
import numpy as np
import librosa  
from functools import lru_cache
import time
import argparse
from espnet2.bin.asr_inference import Speech2Text
from new_conf_words import new_conf_words
import os

@lru_cache
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000)
    return a

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

def transcribe(model,audio):
    nbest = model(audio)
    return nbest[0][0]

def generate_transcription(audio_path,config_file,model_file,device='cuda'):
    os.chdir('/home/suryansh/MADHAV/asr_train_asr_raw_hindi_bpe500')
    speech2text = Speech2Text(config_file,model_file,device=device)
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

    a = load_audio_chunk(audio_path,curr_time,min(curr_time+7, duration))
    txt = initial_audio(speech2text,a)
    words = txt.split()
    conf_words += words[:-4]
    buffer += words[-4:]
    curr_time += 1
    transcription += " "+" ".join(conf_words)
    # curr_time = 1
    while curr_time+7<duration:
        start_time = time.time()
        a = load_audio_chunk(audio_path,curr_time,min(curr_time+7, duration))
        txt = initial_audio(speech2text,a)
        print(txt)
        curr_time += 1
        word_list = txt.split()
        conf_words,buffer,temp = new_conf_words(buffer,word_list,conf_words)
        transcription += " "+" ".join(temp)
        print(curr_time-1, min(curr_time+6, duration), transcription, time.time()-start_time)
        second_count+=1
        delay = second_count - (time.time() - initial_time)
        if delay > 0:
            time.sleep(delay)
        print(second_count)
    transcription += " "+" ".join(buffer)
    return transcription



#Example command:
#python3 ESPnet_streaming.py /home/suryansh/MADHAV/hindi_data/givenFolderHSp/finalSplitAudio/8janA_33.wav /home/suryansh/MADHAV/asr_train_asr_raw_hindi_bpe500/exp/asr_train_asr_raw_hindi_bpe500/config.yaml /home/suryansh/MADHAV/asr_train_asr_raw_hindi_bpe500/exp/asr_train_asr_raw_hindi_bpe500/valid.acc.ave_10best.pth