import sys
import numpy as np
import librosa  
from functools import lru_cache
import time
import argparse
from espnet2.bin.asr_inference import Speech2Text
from new_conf_words import new_conf_words

buffer = []
transcription = ""
conf_words=[]
word_list=[]


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

parser = argparse.ArgumentParser()
parser.add_argument('audio_path', type=str, help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
parser.add_argument('asr_train_config',type=str,help="Filename of the training configuration file extension .yaml")
parser.add_argument('asr_model_file',type=str,help="Filename of the model file extension .pth")
parser.add_argument('--device',type=str,default='cuda',help="Device cpu/cuda default is cuda")
args = parser.parse_args()

logfile = sys.stderr

audio_path = args.audio_path

speech2text = Speech2Text(args.asr_train_config,args.asr_model_file,device=args.device)

SAMPLING_RATE = 16000
audio_len = len(load_audio(audio_path))
duration = audio_len/SAMPLING_RATE
print("Audio duration is: %2.2f seconds" % duration, file=logfile)

curr_time=0
for i in range(0,3):
    a = load_audio_chunk(audio_path,i,i+1)
    txt = initial_audio(speech2text,a)
    transcription += txt
    conf_words.append(txt)

curr_time = 1
while curr_time<audio_len:
    a = load_audio_chunk(audio_path,curr_time,curr_time+3)
    txt = initial_audio(speech2text,a)
    curr_time += 1
    word_list = txt.split()
    conf_words,buffer,temp = new_conf_words(buffer,word_list,conf_words)
    transcription += " ".join(temp)
    print(transcription)
