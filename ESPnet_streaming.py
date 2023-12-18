import sys
import numpy as np
import librosa  
from functools import lru_cache
import time
import argparse

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


def initial_audio():
    global buffer
    global transcription
    text = transcribe(buffer)
    return text



parser = argparse.ArgumentParser()
parser.add_argument('audio_path', type=str, help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
args = parser.parse_args()

logfile = sys.stderr

audio_path = args.audio_path

SAMPLING_RATE = 16000
duration = len(load_audio(audio_path))/SAMPLING_RATE
print("Audio duration is: %2.2f seconds" % duration, file=logfile)

for i in range(0,3):
    a = load_audio_chunk(audio_path,i,i+1)
    buffer.append(a)
    t=initial_audio()
    transcription += t
    word_list.append(t)
