#! python

from nltk.tokenize import word_tokenize

class Tokenizer():
    def __init__(self):
        return

    def mr(self, mr_obj):
        mr_token = []
        for i, attr in enumerate(mr_obj):
            if i > 0:
                mr_token.append('|')
            if mr_obj[attr] == '':
                mr_token.append(mr_obj[attr])
            else:
                a_tmp = word_tokenize(mr_obj[attr])
                for j in range(len(a_tmp)):
                    mr_token.append(a_tmp[j])
        return mr_token

    def txt(self, input_txt):
        return word_tokenize(input_txt)
