def conf_word_check(conf_words, word_list):
    conf_word= conf_words[-1]
    modified_word_list=word_list[4:]
    for i in range(len(modified_word_list)):
        if(conf_word==modified_word_list[i]):
            return i+5
    return 4

# conf_words=['a','b','c','d','e','f','g','h','i','j','k','l','m','n']
# word_list=['k','l','m','n', 'o', 'p', 'q', 'r', 's']
# print(conf_word_check(conf_words, word_list), word_list[conf_word_check(conf_words, word_list)])
    