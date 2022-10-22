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
        mr_idx = [self.dictionary['mr']['s2i']['<sos>']]
        num_token = len(mr_idx)
        for i in range(num_token, self.dictionary['mr']['max_num_token']):
            mr_tensor = torch.LongTensor(mr_idx).unsqueeze(0).to(self.device)
            mr_mask = self.model.make_mr_mask(mr_tensor)
            with torch.no_grad():
                mr_predict, attention = self.model.decoder(enc_txt, txt_mask, mr_tensor, mr_mask)
            mr_predict_idx = mr_predict.argmax(2)[:,-1].item()
            if mr_predict_idx == self.dictionary['mr']['s2i']['<eos>']:
                break
            if i >= self.dictionary['mr']['max_num_token']:
                break
            mr_idx.append(mr_predict_idx)
        # delete <sos>
        mr_idx = mr_idx[1:]

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
            n_order = 1
            for idx in mr_idx:
                attr = self.dictionary['mr']['i2attribute'][idx]
                if attr != 'other':
                    output_mr_obj['value_lex'][attr] = self.dictionary['mr']['i2s'][idx]
                    output_mr_obj['order'][attr] = n_order
                    n_order += 1

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
            for idx in mr_idx:
                attr = self.dictionary['mr']['i2attribute'][idx]
                if attr != 'other':
                    output_mr_obj['value_lex'][attr] = self.dictionary['mr']['i2s'][idx]

        return output_mr_obj, attention
