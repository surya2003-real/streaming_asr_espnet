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

### Example Usage
```python
conf_words=['a','b','c','d','e','f','g','h','i','j','k','l','m','n']
word_list=['k','l','m','n', 'o', 'p', 'q', 'r', 's']

# Call the function and print the result
result_index = conf_word_check(conf_words, word_list)
print(result_index, word_list[result_index])
# Output: 4 o
```
