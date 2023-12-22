# Streaming ASR ESPnet Model

## `generate_transcription` Function

The `generate_transcription` function takes inputs for the model and audio file path

### Arguments
- `audio_path` : The path of the audio file on which we are running ASR
- `config_file` : The model configuration file, this should have an extension _.yaml_
- `model_file` : The model file, this should have an extension _.pth_
- `device` : The device on which we are running the model it can be _cpu/cuda_. The default is _cuda_

We are using the `time` module to simulate live-streaming so that we wait for the next audio chunk to be inputed before processing it. We are using a buffer of length __7__ seconds where we first transcribe the first 7 seconds and then we keep jumping by __1__ second and process the output of that audio file. We take care of ignoring overlaps while appending the text.

- `transcription` : Stores the entire transcription upto the time we have __confidently__ transcribed
- `conf_words` : Stores the words which we are confident are correct
- `buffer_words` : Stores the words that will be added to the new buffer region and on which we need to check for overlaps to ensure confidence

## `conf_word_check` Function

The `conf_word_check` function takes two lists of strings as its arguments and returns an integer value.

### Arguments
- `conf_words`: A list of strings representing a sequence of words.
- `word_list`: Another list of strings representing a sequence of words.

### Output
The function returns an integer value, `k` and an `alarm` flag to .

### Functionality
The function compares the elements of `conf_words` and `word_list` to find the index `k` where the two sequences match. It iterates through the elements of `conf_words` and checks for matches with the corresponding elements in `word_list`. If a match is found, it continues checking the subsequent elements in both sequences until a mismatch is encountered or one of the sequences reaches its end.

## `new_conf_words` Function

The `new_conf_words` function is designed to update the `conf_words` list based on the input parameters. It utilizes the `conf_word_check` function and returns two lists.

### Arguments
- `buffer`: A list of strings representing a buffer sequence.
- `word_list`: A list of strings representing a sequence of words.
- `conf_words`: A list of strings representing the current state of confirmed words.

### Output
The function returns two lists:
- `conf_words`: Updated list of confirmed words.
- `buffer`: Remaining elements in the buffer after processing.
- `transcript_addition`: new words to be added in the transcript

### Functionality
The function first uses the `conf_word_check` function to find the index `k` where the sequences `conf_words` and `word_list` match. It then updates the `conf_words` list by appending elements from the `buffer` until a mismatch is encountered or the buffer is exhausted. The remaining elements in the buffer after processing are returned.
