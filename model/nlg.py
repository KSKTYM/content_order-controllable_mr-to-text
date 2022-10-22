#! python
# -*- coding: utf-8 -*-

import torch
import json
import copy
import sys
sys.path.append('..')
from tokenizer.tokenizer import Tokenizer
from model.nlu import NLU

class NLG():
    def __init__(self, d_model_nlg, f_model_nlg, d_model_nlu, f_model_nlu, algorithm_nlg, algorithm_nlu):
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # NLG model
        saved_data = torch.load(d_model_nlg.rstrip('/')+'/'+f_model_nlg)
        if 'model_nlg' in saved_data:
            self.model_NLG = saved_data['model_nlg']
        elif 'model' in saved_data:
            self.model_NLG = saved_data['model']
        self.model_NLG = self.model_NLG.to(self.device)
        self.model_NLG.eval()

        # beam-search settings
        self.n_beam = 5

        # NLG dictionary
        with open(d_model_nlg.rstrip('/')+'/dictionary.json', 'r', encoding='utf-8') as f:
            self.dictionary = json.load(f)

        # tokenizer
        self.tokenizer = Tokenizer()

        # NLG algorithm
        self.algorithm = algorithm_nlg

        # NLU
        self.NLU = NLU(d_model_nlu, f_model_nlu, algorithm_nlu)

    def _check_mr(self, mr_obj_ref, mr_obj_tgt):
        if self.algorithm == 'A':
            if (mr_obj_ref['value_lex'] == mr_obj_tgt['value_lex']) and \
               (mr_obj_ref['order'] == mr_obj_tgt['order']):
                return True
            else:
                return False
        else:
            if (mr_obj_ref['value_lex'] == mr_obj_tgt['value_lex']):
                return True
            else:
                return False

    def _check_mr_count(self, mr_obj_ref, mr_obj_tgt):
        count = 0
        if self.algorithm == 'A':
            for attr in mr_obj_ref['value_lex']:
                if mr_obj_ref['value_lex'][attr] == mr_obj_tgt['value_lex'][attr]:
                    count += 1
                if mr_obj_ref['order'][attr] == mr_obj_tgt['order'][attr]:
                    count += 1
        else:
            for attr in mr_obj_ref['value_lex']:
                if mr_obj_ref['value_lex'][attr] == mr_obj_tgt['value_lex'][attr]:
                    count += 1
        return count

    def _shape_txt(self, mr_obj, output_txt_token):
        output_txt = ''
        for i in range(len(output_txt_token)):
            if (i > 0) and (output_txt_token[i] != '.') and (output_txt_token[i] != ','):
                if (output_txt_token[i][0] != '\'') and (output_txt_token[i] != 'n\'t'):
                    output_txt += ' '
                else:
                    if (output_txt_token[i] == '\'NAME') or \
                       (output_txt_token[i] == '\'NEAR') or \
                       (output_txt_token[i] == '\'average') or \
                       (output_txt_token[i] == '\'family'):
                        output_txt += ' '
            output_txt += output_txt_token[i]
        output_txt_lex = output_txt

        # lexicalisation
        output_txt = output_txt.replace('NAME', mr_obj['value']['name'])
        output_txt = output_txt.replace('NEAR', mr_obj['value']['near'])

        return output_txt, output_txt_lex

    ## MR-to-Text
    def convert_nlg(self, search_method, mr_obj):
        if ('startword' in mr_obj) is False:
            mr_obj['startword'] = ''

        # MR index <sos> ... <eos>
        mr_idx = [self.dictionary['mr']['s2i']['<sos>']]
        if self.algorithm == 'A':
            for i in range(1, 9):
                for attr in mr_obj['order']:
                    if mr_obj['order'][attr] == i:
                        mr_idx.append(self.dictionary['mr']['s2i'][mr_obj['value_lex'][attr]])
        else:
            for attr in mr_obj['value_lex']:
                if mr_obj['value_lex'][attr] != '':
                    mr_idx.append(self.dictionary['mr']['s2i'][mr_obj['value_lex'][attr]])
        mr_idx.append(self.dictionary['mr']['s2i']['<eos>'])

        # encode
        mr_tensor = torch.LongTensor(mr_idx).unsqueeze(0).to(self.device)
        mr_mask = self.model_NLG.make_mr_mask(mr_tensor)
        enc_mr = self.model_NLG.encoder(mr_tensor, mr_mask)

        # decode
        # greedy search
        output_txt_token, attention_greedy, too_long_flag = self._nlg_decode_greedy(enc_mr, mr_mask, mr_obj)
        output_txt_greedy, output_txt_greedy_lex = self._shape_txt(mr_obj, output_txt_token)
        if search_method == 'greedy':
            return output_txt_greedy, attention_greedy
        else:
            mr_obj_greedy, _ = self.NLU.convert_nlu(output_txt_greedy_lex, txt_token=output_txt_token)
            if (self._check_mr(mr_obj, mr_obj_greedy) is True) and (too_long_flag is False):
                return output_txt_greedy, attention_greedy
            else:
                output_txt_beam, attention_beam, beam_flag, beam_OK_flag = self._nlg_decode_beam(enc_mr, mr_mask, mr_obj)
                if beam_flag is True:
                    return output_txt_beam, attention_beam
                else:
                    if too_long_flag is True:
                        if beam_OK_flag is True:
                            return output_txt_beam, attention_beam
                        else:
                            return output_txt_greedy, attention_greedy
                    else:
                        return output_txt_greedy, attention_greedy

    ## decoder
    # greedy search
    def _nlg_decode_greedy(self, enc_mr, mr_mask, mr_obj):
        a_txt_idx = [self.dictionary['txt']['s2i']['<sos>']]

        # startword
        token_startword = self.tokenizer.txt(mr_obj['startword'])
        for token in token_startword:
            if token in self.dictionary['txt']['s2i']:
                a_txt_idx.append(self.dictionary['txt']['s2i'][token])
            else:
                print('[NLG (decode greedy)] unknown token: '+str(token))
                a_txt_idx.append(self.dictionary['txt']['s2i']['<unk>'])

        too_long_flag = False
        num_token = len(a_txt_idx)
        for i in range(num_token, self.dictionary['txt']['max_num_token']):
            txt_tensor = torch.LongTensor(a_txt_idx).unsqueeze(0).to(self.device)
            txt_mask = self.model_NLG.make_txt_mask(txt_tensor)
            with torch.no_grad():
                txt_predict, attention = self.model_NLG.decoder(enc_mr, mr_mask, txt_tensor, txt_mask)
            txt_predict_idx = txt_predict.argmax(2)[:,-1].item()
            if txt_predict_idx == self.dictionary['txt']['s2i']['<eos>']:
                break
            if i >= self.dictionary['txt']['max_num_token'] - 1:
                print('[NLG (decode greedy)] too long output: '+str(i))
                too_long_flag = True
                break
            a_txt_idx.append(txt_predict_idx)
        txt_tokens = [self.dictionary['txt']['i2s'][i] for i in a_txt_idx]
        txt_tokens = txt_tokens[1:]

        return txt_tokens, attention, too_long_flag

    # beam search
    def _nlg_decode_beam(self, enc_mr, mr_mask, mr_obj):
        a_cand_prev = [{'idx': [self.dictionary['txt']['s2i']['<sos>']], 'val': 1.0}]

        # startword
        token_startword = self.tokenizer.txt(mr_obj['startword'])
        offset = len(token_startword)
        for token in token_startword:
            if token in self.dictionary['txt']['s2i']:
                a_cand_prev[0]['idx'].append(self.dictionary['txt']['s2i'][token])
            else:
                print('[NLG (decode beam)] unknown token: '+str(token))
                a_cand_prev[0]['idx'].append(self.dictionary['txt']['s2i']['<unk>'])

        num_token = len(a_cand_prev[0]['idx'])
        a_out = []
        for i in range(self.dictionary['txt']['max_num_token']-num_token-1):
            a_cand = []
            for j in range(len(a_cand_prev)):
                txt_tensor = torch.LongTensor(a_cand_prev[j]['idx']).unsqueeze(0).to(self.device)
                txt_mask = self.model_NLG.make_txt_mask(txt_tensor)
                with torch.no_grad():
                    txt_predict, attention = self.model_NLG.decoder(enc_mr, mr_mask, txt_tensor, txt_mask)
                txt_predict = torch.softmax(txt_predict, dim=-1)
                for n in range(self.n_beam):
                    a_cand.append(copy.deepcopy(a_cand_prev[j]))
                    idx = (torch.argsort(txt_predict, axis=2)[0, i+offset, -(n+1)]).item()
                    val = txt_predict[0, i+offset, idx].item()
                    a_cand[len(a_cand)-1]['idx'].append(idx)
                    a_cand[len(a_cand)-1]['val'] *= val

            a_cand_sort = sorted(a_cand, key=lambda x:x['val'], reverse=True)
            a_cand_prev = []
            nloop = min(len(a_cand_sort), self.n_beam)
            for j in range(nloop):
                if a_cand_sort[j]['idx'][len(a_cand_sort[j]['idx'])-1] == self.dictionary['txt']['s2i']['<eos>']:
                    a_out.append(a_cand_sort[j])
                    if len(a_out) == self.n_beam:
                        break
                else:
                    a_cand_prev.append(a_cand_sort[j])
            if len(a_out) == self.n_beam:
                break

        if len(a_out) == 0:
            return None, None, False, False

        # check NLU
        flag = False
        a_output_txt = []
        a_attention = []
        a_count = []
        for n in range(len(a_out)):
            txt_tokens = [self.dictionary['txt']['i2s'][i] for i in a_out[n]['idx']]
            txt_tokens = txt_tokens[1:-1]

            output_txt, output_txt_lex = self._shape_txt(mr_obj, txt_tokens)
            mr_obj_beam, _ = self.NLU.convert_nlu(output_txt_lex, txt_token=txt_tokens)
            a_output_txt.append(output_txt)
            a_attention.append(attention)
            if self._check_mr(mr_obj, mr_obj_beam) is True:
                flag = True
                return output_txt, attention, True, True
            else:
                a_count.append(self._check_mr_count(mr_obj, mr_obj_beam))

        if flag is False:
            idx = torch.argmax(torch.tensor(a_count)).item()
            return a_output_txt[idx], a_attention[idx], False, False
