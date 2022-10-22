#! python
# -*- coding: utf-8 -*-

import json

class Dictionary():
    def __init__(self, f_train, f_valid, f_test, tokenizer):
        def _make_value_list_mr(a_obj, value_list):
            for obj in a_obj:
                for attr in obj['mr']['value_lex']:
                    if (obj['mr']['value_lex'][attr] in value_list['mr'][attr]) is False:
                        value_list['mr'][attr].append(obj['mr']['value_lex'][attr])
                if value_list['max_num_sen'] < obj['mr']['num_sen']:
                    value_list['max_num_sen'] = obj['mr']['num_sen']
            return value_list

        def _make_value_list_txt(a_obj, value_list, tokenizer):
            for obj in a_obj:
                token_txt = tokenizer.txt(obj['txt_lex'])
                for token in token_txt:
                    if (token in value_list['txt']) is False:
                        value_list['txt'].append(token)
                if value_list['max_num_token_txt'] < len(token_txt):
                    value_list['max_num_token_txt'] = len(token_txt)
            return value_list

        self.tokenizer = tokenizer

        ##
        # read dataset
        ##
        value_list = {
            'mr': {
                'name': [''],
                'eatType': [''],
                'food': [''],
                'priceRange': [''],
                'customer rating': [''],
                'area': [''],
                'familyFriendly': [''],
                'near': ['']
            },
            'txt': [],
            'max_num_token_txt': 1,
            'max_num_sen': 1
        }

        # train
        with open(f_train, 'r', encoding='utf-8') as f:
            a_obj_train = json.load(f)
        value_list = _make_value_list_mr(a_obj_train, value_list)
        value_list = _make_value_list_txt(a_obj_train, value_list, self.tokenizer)
        del a_obj_train

        # valid
        with open(f_valid, 'r', encoding='utf-8') as f:
            a_obj_valid = json.load(f)
        value_list = _make_value_list_mr(a_obj_valid, value_list)
        value_list = _make_value_list_txt(a_obj_valid, value_list, self.tokenizer)
        del a_obj_valid

        # test
        with open(f_test, 'r', encoding='utf-8') as f:
            a_obj_test = json.load(f)
        value_list = _make_value_list_mr(a_obj_test, value_list)
        value_list = _make_value_list_txt(a_obj_test, value_list, self.tokenizer)
        del a_obj_test

        # sort
        for attr in value_list['mr']:
            value_list['mr'][attr] = sorted(value_list['mr'][attr])
        value_list['txt'] = sorted(value_list['txt'])

        ##
        # dictionary settings
        ##
        self.dictionary = {}

        # TXT
        self.dictionary['txt'] = {}
        self.dictionary['txt']['s2i'] = {}
        self.dictionary['txt']['i2s'] = []
        self.dictionary['txt']['s2i']['<sos>'] = 0
        self.dictionary['txt']['i2s'].append('<sos>')
        self.dictionary['txt']['s2i']['<eos>'] = 1
        self.dictionary['txt']['i2s'].append('<eos>')
        self.dictionary['txt']['s2i']['<pad>'] = 2
        self.dictionary['txt']['i2s'].append('<pad>')
        self.dictionary['txt']['s2i']['<unk>'] = 3
        self.dictionary['txt']['i2s'].append('<unk>')
        offset = len(self.dictionary['txt']['i2s'])
        for i, token in enumerate(value_list['txt']):
            self.dictionary['txt']['s2i'][token] = i+offset
            self.dictionary['txt']['i2s'].append(token)
        self.dictionary['txt']['dim'] = len(self.dictionary['txt']['s2i'])

        self.dictionary['txt']['max_num_token'] = value_list['max_num_token_txt']
        # "+2": <sos> <eos>
        self.dictionary['txt']['max_num_token'] += 2
        # "+10": margin
        self.dictionary['txt']['max_num_token'] += 10
        
        # MR (all)
        self.dictionary['mr'] = {}
        self.dictionary['mr']['s2i'] = {}
        self.dictionary['mr']['i2s'] = []
        self.dictionary['mr']['s2attribute'] = {}
        self.dictionary['mr']['i2attribute'] = []
        self.dictionary['mr']['s2i']['<sos>'] = 0
        self.dictionary['mr']['i2s'].append('<sos>')
        self.dictionary['mr']['s2attribute']['<sos>'] = 'other'
        self.dictionary['mr']['i2attribute'].append('other')
        self.dictionary['mr']['s2i']['<eos>'] = 1
        self.dictionary['mr']['i2s'].append('<eos>')
        self.dictionary['mr']['s2attribute']['<eos>'] = 'other'
        self.dictionary['mr']['i2attribute'].append('other')
        self.dictionary['mr']['s2i']['<pad>'] = 2
        self.dictionary['mr']['i2s'].append('<pad>')
        self.dictionary['mr']['s2attribute']['<pad>'] = 'other'
        self.dictionary['mr']['i2attribute'].append('other')
        self.dictionary['mr']['s2i']['<unk>'] = 3
        self.dictionary['mr']['i2s'].append('<unk>')
        self.dictionary['mr']['s2attribute']['<unk>'] = 'other'
        self.dictionary['mr']['i2attribute'].append('other')
        offset = len(self.dictionary['mr']['i2s'])
        i = 0
        for attr in value_list['mr']:
            for value in value_list['mr'][attr]:
                if (value in self.dictionary['mr']['s2i']) is False:
                    self.dictionary['mr']['s2i'][value] = i+offset
                    self.dictionary['mr']['i2s'].append(value)
                    if value != '':
                        self.dictionary['mr']['s2attribute'][value] = attr
                        self.dictionary['mr']['i2attribute'].append(attr)
                    else:
                        self.dictionary['mr']['s2attribute'][value] = 'other'
                        self.dictionary['mr']['i2attribute'].append('other')
                    i += 1
        self.dictionary['mr']['dim'] = len(self.dictionary['mr']['s2i'])

        # "+2": <sos> <eos>
        self.dictionary['mr']['max_num_token'] = len(value_list['mr']) + 2

    def get_dictionary(self):
        return self.dictionary
