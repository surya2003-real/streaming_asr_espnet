from find_microphones import find_mics
import pyaudio as pa
import pandas as pd
import jiwer
from funasr_onnx import Fsmn_vad
from panns_inference import AudioTagging
from new_conf_words import new_conf_words
import numpy as np
import os
import time
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.enh_inference import SeparateSpeech
import contextlib
from io import StringIO
import sys

@contextlib.contextmanager
def suppress_output():
    # Create a dummy output collector
    dummy_stdout = StringIO()

    # Redirect stdout to the dummy collector
    original_stdout = sys.stdout
    sys.stdout = dummy_stdout

    try:
        yield dummy_stdout
    finally:
        # Reset stdout to its original state
        sys.stdout = original_stdout

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

def transcribe(model,audio):
    nbest = model(audio)
    return nbest[0][0]


def reshape(audio_data):
    if len(audio_data.shape)==1:
        audio_data=audio_data.reshape((audio_data.shape[0],1))
    return audio_data

#Funtions for transcript
def with_transcript(transcript, asr_prediction):
    hyp = jiwer.ReduceToListOfListOfWords()(transcript)
    tru = jiwer.ReduceToListOfListOfWords()(asr_prediction)
    out = jiwer.process_words(asr_prediction, transcript)
    req = out.alignments[0]
    rlen = len(tru[0])
    text_out = []
    for it in req:
        diff = it.ref_start_idx-it.hyp_start_idx
        for ind in range(it.hyp_start_idx, it.hyp_end_idx):
            text_out.append(hyp[0][ind])
            if ind+diff >= rlen:
                break
            index = ind
    sentence = ' '.join(text_out)
    return sentence, index

def live_transcript(transcript, prediction, ref):
    ind = 21
    ref = transcript[ref:ref+ind]
    ref_sen = ' '.join(ref)
    text, it = with_transcript(ref_sen, prediction)
    return text

# Main function
def subtitle_feed(model_dir="asr_train_asr_raw_hindi_bpe500",
                   config_file="exp/asr_train_asr_raw_hindi_bpe500/config.yaml",
                   model_file="exp/asr_train_asr_raw_hindi_bpe500/valid.acc.ave_10best.pth",
                   devices='cuda', min_speech_limit=0.1, music_tolerance=0.5, frame_length=7, 
                   SAMPLE_RATE=16000,CHANNELS=1,CHUNK=1000, vad_on=True,enh_on=True,
                   transcript_path = None):

    os.chdir(model_dir)
    model = Speech2Text(config_file,model_file,device=devices)
    dev_idx = find_mics()
    p = pa.PyAudio()

    if(enh_on):
        separate_speech = {}
        enh_model_sc = SeparateSpeech(
            train_config="../speech_sc_enhance/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/config.yaml",
            model_file="../speech_sc_enhance/enh_model_sc/exp/enh_train_enh_conv_tasnet_raw/5epoch.pth",
            # for segment-wise process on long speech
            normalize_segment_scale=False,
            show_progressbar=True,
            device= devices,
            ref_channel=1,
            normalize_output_wav=True,
        )
    at = AudioTagging(checkpoint_path=None,device=devices)

    #Variable initializations
    buffer = []
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
    further = True
    try:
        for _ in range(0, int(SAMPLE_RATE / CHUNK * frame_length)):
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        further = False

    # time.sleep(7)
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_data = pcm2float(audio_data)

    # Speech Enhancement
    if(enh_on):
        enhanced = reshape(audio_data)
        mixwav_sc = enhanced[:,0]
        wave = mixwav_sc[None, :]
        with suppress_output() as output_collector:
            (clipwise_output, embedding) = at.inference(wave)
        # print(clipwise_output[0][137])
        if clipwise_output[0][0]>min_speech_limit:
            if clipwise_output[0][137]>=music_tolerance:
                # print('Enhancement Required')
                wave = enh_model_sc(mixwav_sc[None, ...], SAMPLE_RATE)
        audio_data=wave[0].squeeze()


    txt = transcribe(model,audio_data)

    # when transcript is available
    if not transcript_path is None:
        ref = 0
        with open(transcript_path, 'r' ) as file:
            trans = file.read()
        transcript = jiwer.ReduceToListOfListOfWords()(trans)[0]
        txt = live_transcript(transcript,txt,ref)

    words = txt.split()
    conf_words += words[:-4]
    buffer += words[-4:]
    curr_time += 1
    transcription += " "+" ".join(conf_words)
    print(" "+" ".join(conf_words), end="")
    # print(curr_time-1, curr_time-1+frame_length, transcription)

    # Keep the program running
    while further:
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

            # Speech Enhancement
            if(enh_on):
                enhanced = reshape(new_data)
                mixwav_sc = enhanced[:,0]
                wave = mixwav_sc[None, :]  # (batch_size, segment_samples)
                with suppress_output() as output_collector:
                    (clipwise_output, embedding) = at.inference(wave)
                # print(clipwise_output[0][137])
                # print(clipwise_output[0][0])
                if clipwise_output[0][0]>min_speech_limit:
                        if clipwise_output[0][137]>=music_tolerance:
                            # print('Enhancement Required')
                            wave = enh_model_sc(mixwav_sc[None, ...], SAMPLE_RATE)
                new_data=wave[0].squeeze()
            
            # VAD
            if(vad_on):
                model_dir = "../FSMN-VAD"
                vad_model = Fsmn_vad(model_dir,quantize=True)
                result = vad_model(new_data)
                result = np.asarray(result, dtype=np.int32)
                result=result.reshape((result.shape[1],2))
                df=pd.DataFrame(result, columns=['start', 'end'])
                # print(df)
                speech = np.array([])
                duration=1*1000
                for _, row in df.iterrows():
                    start_sample = row['start']
                    end_sample = row['end']
                    # print(start_sample, end_sample, duration)
                    if(start_sample<0 and end_sample<0):
                        continue
                    if(start_sample>duration and end_sample>duration):
                        break
                    # print("Y")
                    speech = np.concatenate([speech, new_data[int(max(0,start_sample))*16:int(min(duration, end_sample))*16]])
                # print(speech)
                new_data = speech

            audio_data=np.append(audio_data,new_data)
            # print("audio_data: ", audio_data)
            transcription_data = audio_data[-SAMPLE_RATE*frame_length:]

            start_time = time.time()
            # if a is None:
            #     print("break here")
            #     break
            txt = transcribe(model,transcription_data)

            # with transcript
            if not transcript_path is None:
                txt = live_transcript(transcript,txt,ref)
            
            # print(txt)
            curr_time += 1
            word_list = txt.split()
            # print(word_list)
            word_list=word_list[:-1]
            conf_words,buffer,temp = new_conf_words(buffer,word_list,conf_words)
            if(len(temp)>0):
                transcription += " "+" ".join(temp)
                print(" "+" ".join(temp), end="")
            # print(curr_time-1, curr_time-1+frame_length, transcription, time.time()-start_time)
        except KeyboardInterrupt:
            break

    # Stop stream
    stream.stop_stream()
    stream.close()

    # Terminate PyAudio
    p.terminate()
    print("stream stopped")

    transcription += " "+" ".join(buffer)
    # print(transcription)
    os.chdir("..")
    return transcription
