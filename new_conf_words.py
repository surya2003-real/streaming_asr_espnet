from conf_word_check import conf_word_check
def new_conf_words(buffer, word_list, conf_words):
    k, alarm=conf_word_check(conf_words, word_list)
    k+=alarm
    ind = -max(alarm,4)
    if(len(word_list)==k):
        return conf_words[ind:], [], []
    cnt=0
    for r in range(k,min(len(word_list), k+len(buffer))):
        if(word_list[r]==buffer[cnt]):
            cnt+=1
        else:
            break
    if cnt==0:
        cnt+=1
    return conf_words[ind:]+word_list[k:k+cnt], word_list[k+cnt:], word_list[k:k+cnt]

# conf_words=['l','m','n']
# buffer=['l', 'q']
# word_list=[ 'm', 'n', 'k']
# conf_words, buffer, transcript_addition=new_conf_words(buffer, word_list, conf_words)
# print(conf_words, buffer, transcript_addition)
