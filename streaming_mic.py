from ESPnet_streaming import transcribe
from espnet2.bin.asr_inference import Speech2Text
from new_conf_words import new_conf_words
import librosa
import numpy as np
import os
import sys
import time

# server options
host = 'localhost'
port = 43007
SAMPLING_RATE = 16000

# model options
config_file = "/home/suryansh/MADHAV/asr_train_asr_raw_hindi_bpe500/exp/asr_train_asr_raw_hindi_bpe500/config.yaml"
model_file = "/home/suryansh/MADHAV/asr_train_asr_raw_hindi_bpe500/exp/asr_train_asr_raw_hindi_bpe500/valid.acc.ave_10best.pth"
device = 'cuda'
os.chdir('/home/suryansh/MADHAV/asr_train_asr_raw_hindi_bpe500')
model = Speech2Text(config_file,model_file,device=device)

######### Server objects

import line_packet
import socket

import logging


class Connection:
    '''it wraps conn object'''
    PACKET_SIZE = 65536

    def __init__(self, conn):
        self.conn = conn
        self.last_line = ""

        self.conn.setblocking(True)

    def send(self, line):
        '''it doesn't send the same line twice, because it was problematic in online-text-flow-events'''
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        in_line = line_packet.receive_lines(self.conn)
        return in_line

    def non_blocking_receive_audio(self):
        r = self.conn.recv(self.PACKET_SIZE)
        return r
    
import io
import soundfile

# wraps socket and ASR object, and serves one client connection. 
# next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, c:Connection, min_chunk):
        self.connection = c
        self.min_chunk = min_chunk

        self.last_end = None

    def receive_audio_chunk(self):
        # receive all audio that is available by this time
        # blocks operation if less than self.min_chunk seconds is available
        # unblocks if connection is closed or a chunk is available
        out = []
        while sum(len(x) for x in out) < self.min_chunk*SAMPLING_RATE:
            raw_bytes = self.connection.non_blocking_receive_audio()
            print(raw_bytes[:10])
            print(len(raw_bytes))
            if not raw_bytes:
                break
            sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
            audio, _ = librosa.load(sf,sr=SAMPLING_RATE)
            out.append(audio)
        if not out:
            return None
        return np.concatenate(out)

    # def format_output_transcript(self,o):
    #     # output format in stdout is like:
    #     # 0 1720 Takhle to je
    #     # - the first two words are:
    #     #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
    #     # - the next words: segment transcript

    #     # This function differs from whisper_online.output_transcript in the following:
    #     # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
    #     # Therefore, beg, is max of previous end and current beg outputed by Whisper.
    #     # Usually it differs negligibly, by appx 20 ms.

    #     if o[0] is not None:
    #         beg, end = o[0]*1000,o[1]*1000
    #         if self.last_end is not None:
    #             beg = max(beg, self.last_end)

    #         self.last_end = end
    #         print("%1.0f %1.0f %s" % (beg,end,o[2]),flush=True,file=sys.stderr)
    #         return "%1.0f %1.0f %s" % (beg,end,o[2])
    #     else:
    #         print(o,file=sys.stderr,flush=True)
    #         return None

    # def send_result(self, prev_transcript, transcription):
    #     msg = self.format_output_transcript(prev_transcript, transcription)
    #     if msg is not None:
    #         self.connection.send(msg)

    def process(self):
        # handle one client connection

        #Variable initializations
        buffer = []
        prev_transcript = ""
        transcription = ""
        conf_words=[]
        word_list=[]

        time.sleep(7)

        curr_time = 0
        initial_time = time.time()
        a = self.receive_audio_chunk()
        txt = transcribe(model,a)
        words = txt.split()
        conf_words += words[:-4]
        buffer += words[-4:]
        curr_time += 1
        transcription += " "+" ".join(conf_words)
        print(curr_time-1, curr_time+6, transcription)

        try:
            while True:
                if(1-(time.time()-initial_time)>0):
                    time.sleep(1-(time.time()-initial_time))
                start_time = time.time()
                a = self.receive_audio_chunk()
                if a is None:
                    print("break here",file=sys.stderr)
                    break
                txt = transcribe(model,a)
                print(txt)
                curr_time += 1
                word_list = txt.split()
                print(word_list)
                word_list=word_list[:-1]
                conf_words,buffer,temp = new_conf_words(buffer,word_list,conf_words)
                if(len(temp)>0):
                    transcription += " "+" ".join(temp)
                print(curr_time-1, curr_time+6, transcription, time.time()-start_time)
                initial_time = time.time()
                # self.online_asr_proc.insert_audio_chunk(a)
                # o = online.process_iter()
        except KeyboardInterrupt:
            print("broken pipe -- connection closed?")
        transcription += " "+" ".join(buffer)
        print(transcription)
        return transcription

#        o = online.finish()  # this should be working
#        self.send_result(o)
    
level = logging.INFO
# logging.basicConfig(level=level, format='whisper-server-%(levelname)s: %(message)s')

# server loop

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    print('INFO: Listening on'+str((host, port)))
    while True:
        conn, addr = s.accept()
        print('INFO: Connected to client on {}'.format(addr))
        connection = Connection(conn)
        proc = ServerProcessor(connection, 8)
        proc.process()
        conn.close()
        logging.info('INFO: Connection to client closed')
        break
print('INFO: Connection closed, terminating.')
