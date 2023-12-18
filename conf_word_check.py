def conf_word_check(conf_words, word_list):
    k=0
    i=0
    while(i < len(conf_words) and k < len(word_list)):
        if(conf_words[i]==word_list[k]):
            flag=1
            temp1=i
            temp2=k
            while(temp1<len(conf_words)-1 and temp2<len(word_list)-1):
                temp1+=1
                temp2+=1
                if (conf_words[temp1]!=word_list[temp2]):
                    flag=0
                    break
            if(flag):
                i=temp1+1
                k=temp2+1
        i+=1
    return k

# conf_words=['a','b','c','d','e','f','g','h','i','j','k','l','m','n']
# word_list=['k','l','m','n', 'o', 'p', 'q', 'r', 's']
# print(conf_word_check(conf_words, word_list), word_list[conf_word_check(conf_words, word_list)])
    