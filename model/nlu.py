#! python
# -*- coding: utf-8 -*-

import torch
import json
import copy
import sys
sys.path.append('..')
from tokenizer.tokenizer import Tokenizer

class NLU():
    def __init__(self, d_model, f_model, algorithm):
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # NLU model
        saved_data = torch.load(d_model.rstrip('/')+'/'+f_model)
        if 'model_nlu' in saved_data:
            self.model = saved_data['model_nlu']
        elif 'model' in saved_data:
            self.model = saved_data['model']
        self.model = self.model.to(self.device)
        self.model.eval()

        # NLU dictionary
        with open(d_model.rstrip('/')+'/dictionary.json', 'r', encoding='utf-8') as f:
            self.dictionary = json.load(f)

        # tokenizer
        self.tokenizer = Tokenizer()

        # NLU algorithm
        self.algorithm = algorithm

    def convert_nlu(self, input_txt, txt_token=None):
        # tokenize
        if txt_token is None:
            txt_token = self.tokenizer.txt(input_txt)
        a_txt_idx = [self.dictionary['txt']['s2i']['<sos>']]
        for token in txt_token:
            if token in self.dictionary['txt']['s2i']:
                a_txt_idx.append(self.dictionary['txt']['s2i'][token])
            else:
                print('[NLU] unknown token: '+str(token))
                a_txt_idx.append(self.dictionary['txt']['s2i']['<unk>'])
        a_txt_idx.append(self.dictionary['txt']['s2i']['<eos>'])
        txt_tensor = torch.LongTensor(a_txt_idx).unsqueeze(0).to(self.device)

        # encode
        txt_mask = self.model.make_txt_mask(txt_tensor)
        with torch.no_grad():
            enc_txt = self.model.encoder(txt_tensor, txt_mask)

        # decode
        mr_value_idx = [self.dictionary['mr_value']['s2i']['<sos>']]
        num_token = len(mr_value_idx)
        for i in range(num_token, self.dictionary['mr_value']['max_num_token']):
            mr_value_tensor = torch.LongTensor(mr_value_idx).unsqueeze(0).to(self.device)
            mr_mask = self.model.make_mr_mask(mr_value_tensor)
            with torch.no_grad():
                mr_value_predict, attention = self.model.decoder(enc_txt, txt_mask, mr_value_tensor, mr_mask)
            mr_value_predict_idx = mr_value_predict.argmax(2)[:,-1].item()
            if mr_value_predict_idx == self.dictionary['mr_value']['s2i']['<eos>']:
                break
            if i >= self.dictionary['mr_value']['max_num_token']:
                break
            mr_value_idx.append(mr_value_predict_idx)
        mr_value = [self.dictionary['mr_value']['i2s'][i] for i in mr_value_idx]
        mr_value = mr_value[1:]

        if self.algorithm == 'A':
            output_mr_obj = {
                'value_lex': {
                    'name': '',
                    'eatType': '',
                    'food': '',
                    'priceRange': '',
                    'customer rating': '',
                    'area': '',
                    'familyFriendly': '',
                    'near': ''
                },
                'order': {
                    'name': 0,
                    'eatType': 0,
                    'food': 0,
                    'priceRange': 0,
                    'customer rating': 0,
                    'area': 0,
                    'familyFriendly': 0,
                    'near': 0
                }
            }
            j = 1
            for i in range(len(mr_value)):            
                if self.dictionary['mr_value']['s2attribute'][mr_value[i]] != 'other':
                    output_mr_obj['value_lex'][self.dictionary['mr_value']['s2attribute'][mr_value[i]]] = mr_value[i]
                    output_mr_obj['order'][self.dictionary['mr_value']['s2attribute'][mr_value[i]]] = j
                    j += 1

        else:
            output_mr_obj = {
                'value_lex': {
                    'name': '',
                    'eatType': '',
                    'food': '',
                    'priceRange': '',
                    'customer rating': '',
                    'area': '',
                    'familyFriendly': '',
                    'near': ''
                }
            }
            for i in range(len(mr_value)):            
                if self.dictionary['mr_value']['s2attribute'][mr_value[i]] != 'other':
                    output_mr_obj['value_lex'][self.dictionary['mr_value']['s2attribute'][mr_value[i]]] = mr_value[i]

        return output_mr_obj, attention
