import ESPnet_streaming as espstream
from torchmetrics.functional import word_error_rate,char_error_rate
import argparse
import sys
import os
from espnet2.bin.asr_inference import Speech2Text


parser = argparse.ArgumentParser()
parser.add_argument('asr_train_config',type=str,help="Filename of the training configuration file extension .yaml")
parser.add_argument('asr_model_file',type=str,help="Filename of the model file extension .pth")
parser.add_argument('--device',type=str,default='cuda',help="Device cpu/cuda default is cuda")
args = parser.parse_args()

logfile = sys.stderr

os.chdir('/home/suryansh/MADHAV/asr_train_asr_raw_hindi_bpe500')

speech2text = Speech2Text(args.asr_train_config,args.asr_model_file,device=args.device)

# Add audio paths and corresponding ground_truths according to the data you have
audio_paths = []
ground_truths = []

transcriptions = []

for audio in audio_paths:
    transcriptions.append(espstream.subtitle_audio_file(audio_path=audio, model=speech2text,logfile=logfile))

WER = word_error_rate(preds=transcriptions, target=ground_truths)
CER = char_error_rate(preds=transcriptions, target=ground_truths)

print(f"Word error rate is {WER}")
print(f"Character error rate is {CER}")
