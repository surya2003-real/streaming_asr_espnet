# Streaming ASR ESPnet

## `initial_audio` Function

This is a function that will be called to transcribe the first few words so that we can establish a buffer

## `conf_word_check` Function

The `conf_word_check` function takes two lists of strings as its arguments and returns an integer value.

### Arguments
- `conf_words`: A list of strings representing a sequence of words.
- `word_list`: Another list of strings representing a sequence of words.

### Output
The function returns an integer value, `k`.

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

## `evaluate.py`

Hardcode the list of audio_file paths and their corresponding correct transcriptions.

Then you can pass the model with the config(.yaml) and model_file(.pth) as agruments from the command line.

It will print the WER and CER of the final transcription evalueated over all the audio files
