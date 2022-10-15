#! python
# -*- coding: utf-8 -*-

import json
import random
import torch

# value (variable length)
class MyDataset_O(torch.utils.data.Dataset):
    def __init__(self, fname, dictionary, tokenizer, seed):
        with open(fname, 'r', encoding='utf-8') as f:
            obj_in = json.load(f)

        self.data = obj_in
        self.dictionary = dictionary
        self.tokenizer = tokenizer

        # index
        random.seed(seed)
        self.a_idx = random.sample(range(len(self.data)), k=len(self.data))

    def __len__(self):
        return len(self.data)

    def _mrobj2seq(self, mr_value):
        a_seq = []
        for attr in mr_value:
            if mr_value[attr] != '':
                a_seq.append(mr_value[attr])
        return a_seq

    def __getitem__(self, idx_org):
        idx = self.a_idx[idx_org]

        # MR (value)
        token_mr_value = self._mrobj2seq(self.data[idx]['mr']['value_lex'])
        a_mr_value = [self.dictionary['mr_value']['s2i']['<pad>']] * self.dictionary['mr_value']['max_num_token']
        a_mr_value[0] = self.dictionary['mr_value']['s2i']['<sos>']
        for i in range(len(token_mr_value)):
            a_mr_value[i+1] = self.dictionary['mr_value']['s2i'][token_mr_value[i]]
        a_mr_value[len(token_mr_value)+1] = self.dictionary['mr_value']['s2i']['<eos>']

        # TXT
        token_txt = self.tokenizer.txt(self.data[idx]['txt_lex'])
        a_txt = [self.dictionary['txt']['s2i']['<pad>']] * self.dictionary['txt']['max_num_token']
        a_txt[0] = self.dictionary['txt']['s2i']['<sos>']
        for i in range(len(token_txt)):
            a_txt[i+1] = self.dictionary['txt']['s2i'][token_txt[i]]
        a_txt[len(token_txt)+1] = self.dictionary['txt']['s2i']['<eos>']

        a_mr_value = torch.tensor(a_mr_value)
        a_txt = torch.tensor(a_txt)

        return a_mr_value, a_txt


# value + order (variable length)
class MyDataset_A(torch.utils.data.Dataset):
    def __init__(self, fname, dictionary, tokenizer, seed):
        with open(fname, 'r', encoding='utf-8') as f:
            obj_in = json.load(f)

        self.data = obj_in
        self.dictionary = dictionary
        self.tokenizer = tokenizer

        # index
        random.seed(seed)
        self.a_idx = random.sample(range(len(self.data)), k=len(self.data))

    def __len__(self):
        return len(self.data)

    def _mrobj2seq(self, mr):
        a_seq = []
        for i in range(1, 9):
            for attr in mr['order']:
                if mr['order'][attr] == i:
                    a_seq.append(mr['value_lex'][attr])
        return a_seq

    def __getitem__(self, idx_org):
        idx = self.a_idx[idx_org]

        # MR (value)
        token_mr_value = self._mrobj2seq(self.data[idx]['mr'])
        a_mr_value = [self.dictionary['mr_value']['s2i']['<pad>']] * self.dictionary['mr_value']['max_num_token']
        a_mr_value[0] = self.dictionary['mr_value']['s2i']['<sos>']
        for i in range(len(token_mr_value)):
            a_mr_value[i+1] = self.dictionary['mr_value']['s2i'][token_mr_value[i]]
        a_mr_value[len(token_mr_value)+1] = self.dictionary['mr_value']['s2i']['<eos>']

        # TXT
        token_txt = self.tokenizer.txt(self.data[idx]['txt_lex'])
        a_txt = [self.dictionary['txt']['s2i']['<pad>']] * self.dictionary['txt']['max_num_token']
        a_txt[0] = self.dictionary['txt']['s2i']['<sos>']
        for i in range(len(token_txt)):
            a_txt[i+1] = self.dictionary['txt']['s2i'][token_txt[i]]
        a_txt[len(token_txt)+1] = self.dictionary['txt']['s2i']['<eos>']

        a_mr_value = torch.tensor(a_mr_value)
        a_txt = torch.tensor(a_txt)

        return a_mr_value, a_txt
