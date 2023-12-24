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
from funasr_onnx import Fsmn_vad
from panns_inference import AudioTagging
# Function to load audio file
@lru_cache
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000)
    if len(a.shape)==1:
        a=a.reshape((a.shape[0],1))
    return a

# Function to load an audio chunk from the larger file
def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s, :]

# Function to transcribe audio chunk
def transcribe(model,audio):
    nbest = model(audio)
    return nbest[0][0]

def generate_transcription(audio_path,config_file,model_file,device='cuda', min_speech_limit=0.1, music_tolerance=0.5, frame_length=7, SAMPLING_RATE=16000, VAD_on=True):

    os.chdir('/home/suryansh/MADHAV')
    from espnet2.bin.enh_inference import SeparateSpeech
    separate_speech = {}
    enh_model_sc = SeparateSpeech(
        train_config="./speech_sc_enhance/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/config.yaml",
        model_file="./speech_sc_enhance/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/5epoch.pth",
        # for segment-wise process on long speech
        normalize_segment_scale=False,
        show_progressbar=True,
        ref_channel=1,
        normalize_output_wav=True,
        device="cuda:0",
    )
    os.chdir('/home/suryansh/MADHAV/asr_train_asr_raw_hindi_bpe500')
    speech2text = Speech2Text(config_file,model_file,device=device)
    #Variable initializations
    buffer = []
    transcription = ""
    conf_words=[]
    word_list=[]
    audio_len = len(load_audio(audio_path))
    audio_duration = audio_len/SAMPLING_RATE
    print("Audio duration is: %2.2f seconds" % audio_duration)

    initial_time = time.time()
    curr_time=0
    second_count = 0

    if audio_duration<8:
        a = load_audio_chunk(audio_path,0,audio_duration)
        mixwav_sc = a[:,0]
        wave = mixwav_sc[None, :]  # (batch_size, segment_samples)
        at = AudioTagging(checkpoint_path=None, device='cuda')
        (clipwise_output, embedding) = at.inference(wave)
        print(clipwise_output[0][137])
        if clipwise_output[0][0]>min_speech_limit:
            if clipwise_output[0][137]>=music_tolerance:
                print('Enhancement Required')
                wave = enh_model_sc(mixwav_sc[None, ...], 16000)
        a=wave[0].squeeze()
        txt = transcribe(speech2text,a)
        transcription += " "+txt
        return transcription
    
    # The first 7 seconds of the audio file are transcribed
    a = load_audio_chunk(audio_path,curr_time,min(curr_time+frame_length, audio_duration))
    mixwav_sc = a[:,0]
    wave = mixwav_sc[None, :]  # (batch_size, segment_samples)
    at = AudioTagging(checkpoint_path=None, device='cuda')
    (clipwise_output, embedding) = at.inference(wave)
    print(clipwise_output[0][137])
    print(clipwise_output[0][0])
    if clipwise_output[0][0]>min_speech_limit:
            if clipwise_output[0][137]>=music_tolerance:
                print('Enhancement Required')
                wave = enh_model_sc(mixwav_sc[None, ...], 16000)
    a=wave[0].squeeze()
    txt = transcribe(speech2text,a)
    words = txt.split()
    conf_words += words[:-4]
    buffer += words[-4:]
    curr_time += 1
    transcription += " "+" ".join(conf_words)
    # curr_time = 1
    print(curr_time-1, min(curr_time+frame_length-1, audio_duration), transcription)
    print("breakdown",conf_words, buffer)
    speech = np.array([])
    while curr_time+frame_length<audio_duration:
        start_time = time.time()
        a = load_audio_chunk(audio_path,curr_time,min(curr_time+frame_length, audio_duration))
        mixwav_sc = a[:,0]
        wave = mixwav_sc[None, :]  # (batch_size, segment_samples)
        at = AudioTagging(checkpoint_path=None, device='cuda')
        (clipwise_output, embedding) = at.inference(wave)
        print(clipwise_output[0][137])
        print(clipwise_output[0][0])
        if clipwise_output[0][0]>min_speech_limit:
            if clipwise_output[0][137]>=music_tolerance:
                print('Enhancement Required')
                wave = enh_model_sc(mixwav_sc[None, ...], 16000)
        a=wave[0].squeeze()
        if(VAD_on):
            model_dir = "/home/suryansh/MADHAV/FSMN-VAD"
            model = Fsmn_vad(model_dir, quantize=True)
            result = model(a)
            result = np.asarray(result, dtype=np.int32)
            result=result.reshape((result.shape[1],2))
            df=pd.DataFrame(result, columns=['start', 'end'])
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
            if(len(speech)>500):
                txt = transcribe(speech2text,speech)
                print(txt)
                curr_time += 1
                word_list = txt.split()
                print(word_list)
                # word_list=word_list[:-1]
                conf_words,buffer,temp = new_conf_words(buffer,word_list,conf_words)
                if(len(temp)>0):
                    transcription += " "+" ".join(temp)
                print(curr_time-1, min(curr_time-1+frame_length, audio_duration), transcription, time.time()-start_time)
                print("breakdown",conf_words, buffer)
        else:
            txt = transcribe(speech2text,a)
            print(txt)
            curr_time += 1
            word_list = txt.split()
            print(word_list)
            word_list=word_list[:-1]
            conf_words,buffer,temp = new_conf_words(buffer,word_list,conf_words)
            if(len(temp)>0):
                transcription += " "+" ".join(temp)
            print(curr_time-1, min(curr_time-1+frame_length, audio_duration), transcription, time.time()-start_time)
            print("breakdown",conf_words, buffer)
        second_count+=1
        delay = second_count - (time.time() - initial_time)
        if delay > 0:
            time.sleep(delay)
        print(second_count)
    transcription += " "+" ".join(buffer)
    return transcription
