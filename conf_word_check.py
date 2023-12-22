def conf_word_check(conf_words, word_list):
    print(len(conf_words), len(word_list))
    conf_word= conf_words[-1]
    print(conf_word)
    for l in range(len(conf_words)):
        for i in range(len(word_list)-1,0,-1):
            if(conf_word==word_list[i]):
                return i+1, l
        if(l==len(conf_words)-1):
            return 0, len(conf_words)-1
        conf_word=conf_words[-2-l]

# conf_words=['a','b','c','d','e','f','g','h','i','j','k','l','m','n']
# word_list=['k','l','m','n', 'o', 'p', 'q', 'r', 's']
# print(conf_word_check(conf_words, word_list), word_list[conf_word_check(conf_words, word_list)])
    