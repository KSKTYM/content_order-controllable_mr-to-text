#! python
# -*- coding: utf-8 -*-

import sys
import json
import random
import argparse
sys.path.append('../../..')
from model.nlg import NLG
from model.nlu import NLU

def conv_text2mr(text):
    mr = {}
    mr['name'] = text.split('|')[0]
    mr['eatType'] = text.split('|')[1]
    mr['food'] = text.split('|')[2]
    mr['priceRange'] = text.split('|')[3]
    mr['customer rating'] = text.split('|')[4]
    mr['area'] = text.split('|')[5]
    mr['familyFriendly'] = text.split('|')[6]
    mr['near'] = text.split('|')[7]
    return mr

def remove_duplication(a_obj_in):
    a_idx = []
    a_tmp = []
    for i, obj in enumerate(a_obj_in):
        if (obj in a_tmp) is False:
            a_tmp.append(obj)
            a_idx.append(i)
    a_obj_out = []
    for idx in a_idx:
        a_obj_out.append(a_obj_in[idx])
    return a_obj_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_train', help='training data json file', default='../e2e_train.json')
    parser.add_argument('-d_model_nlg', help='NLG parameter directory', default='../../output/NLG_order')
    parser.add_argument('-d_model_nlu', help='NLU parameter directory', default='../../output/NLU_order')
    parser.add_argument('-f_model_nlg', help='NLG model file', default='best.dat')
    parser.add_argument('-f_model_nlu', help='NLU model file', default='best.dat')
    parser.add_argument('-search', help='NLG search', choices=['best', 'greedy'], default='best')
    parser.add_argument('-alg_nlg', help='O: value, A: value+order', default='A')
    parser.add_argument('-alg_nlu', help='O: value, A: value+order', default='A')
    parser.add_argument('-seed', help='random seed', type=int, default=1234)
    parser.add_argument('-nv', help='number of random variation', type=int, default=16)
    parser.add_argument('-d_data', help='data directory')
    args = parser.parse_args()

    print('** generate extended TXT data **')
    print(' original corpus data')
    print('  training        : '+str(args.f_train)) 
    print(' NLG')
    print('  model           : '+str(args.d_model_nlg.rstrip('/'))+'/'+str(args.f_model_nlg))
    print('  search          : '+str(args.search))
    print('  algorithm       : '+str(args.alg_nlg))
    print(' NLU')
    print('  model           : '+str(args.d_model_nlu.rstrip('/'))+'/'+str(args.f_model_nlu))
    print('  algorithm       : '+str(args.alg_nlu))
    print(' seed             : '+str(args.seed))
    print(' random variation : '+str(args.nv))
    print(' data directory   : '+str(args.d_data))

    with open(args.f_train, 'r', encoding='utf-8') as f:
        a_train = json.load(f)
    a_nea = []
    for i in range(9):
        a_nea.append(0)
    for data in a_train:
        nea = 0
        for attr in data['mr']['value']:
            if data['mr']['value'][attr] != '':
                nea += 1
        a_nea[nea] += 1
    max_nea = max(a_nea)
    print(a_nea)
    print(str(max_nea))
    del a_train

    ##
    ## generate extended TXT data
    ##
    max_number = max_nea * 2
    random.seed(args.seed)

    # NLG model
    NLG_model = NLG(args.d_model_nlg, args.f_model_nlg, args.d_model_nlu, args.f_model_nlu, args.alg_nlg, args.alg_nlu)

    # NLU model
    NLU_model = NLU(args.d_model_nlu, args.f_model_nlu, args.alg_nlu)

    # generate 'extended TXT' data from 'extended MR' data
    with open(args.d_data.rstrip('/')+'/mr_combination_'+str(args.nv)+'.json', 'r', encoding='utf-8') as f:
        a_obj = json.load(f)

    for nea in range(8, 0, -1):
        a_ext_data_ok = []
        a_ext_data_ng = []
        idx_ok = 0
        idx_ng_value = 0
        idx_ng_order = 0
        idx_ng_both = 0

        num_obj = len(a_obj[str(nea)])
        a_idx = random.sample(range(num_obj), num_obj)
        for i in range(num_obj):
            print(str(nea)+'/'+str(i)+'/'+str(num_obj)+'/'+str(max_number)+': '+str(idx_ok)+'/'+str(idx_ng_value)+'/'+str(idx_ng_order)+'/'+str(idx_ng_both))
            obj = {'mr': {'value': a_obj[str(nea)][a_idx[i]]['value_lex'], 'value_lex': a_obj[str(nea)][a_idx[i]]['value_lex'], 'order': a_obj[str(nea)][a_idx[i]]['order']}}
            obj['startword'] = '' 
            txt_predict, _ = NLG_model.convert_nlg(args.search, obj['mr'])
            nlu_mr_obj, _ = NLU_model.convert_nlu(txt_predict)

            ng_value = False
            ng_order = False
            for attr in obj['mr']['value_lex']:
                if (nlu_mr_obj['value_lex'][attr] != obj['mr']['value_lex'][attr]):
                    ng_value = True
                if (nlu_mr_obj['order'][attr] != obj['mr']['order'][attr]):
                    ng_order = True

            obj['txt_lex'] = txt_predict
            del obj['startword']

            if (ng_value is False) and (ng_order is False):
                obj['id'] = 'ex-'+str(nea)+str(idx_ok).zfill(5)
                idx_ok += 1
                a_ext_data_ok.append(obj)
            elif (ng_value is True) and (ng_order is False):
                idx_ng_value += 1
                a_ext_data_ng.append(obj)
            elif (ng_value is False) and (ng_order is True):
                idx_ng_order += 1
                a_ext_data_ng.append(obj)
            else:
                idx_ng_both += 1
                a_ext_data_ng.append(obj)

            #if idx_ok >= max_number:
            #    break
            if idx_ok + a_nea[nea] >= max_number:
                break

        with open(args.d_data.rstrip('/')+'/ok_'+str(args.nv)+'_'+str(nea)+'.json', 'w', encoding='utf-8') as f:
            json.dump(a_ext_data_ok, f, ensure_ascii=False, indent=4, sort_keys=False)
            print('extend data(ok): '+str(len(a_ext_data_ok)))
        with open(args.d_data.rstrip('/')+'/ng_'+str(args.nv)+'_'+str(nea)+'.json', 'w', encoding='utf-8') as f:
            json.dump(a_ext_data_ng, f, ensure_ascii=False, indent=4, sort_keys=False)
            print('extend data(ng): '+str(len(a_ext_data_ng)))

    with open(args.d_data.rstrip('/')+'/ok_'+str(args.nv)+'.json', 'w', encoding='utf-8') as f:
        json.dump(a_ext_data_ok, f, ensure_ascii=False, indent=4, sort_keys=False)
        print('extend data(ok): '+str(len(a_ext_data_ok)))
    with open(args.d_data.rstrip('/')+'/ng_'+str(args.nv)+'.json', 'w', encoding='utf-8') as f:
        json.dump(a_ext_data_ng, f, ensure_ascii=False, indent=4, sort_keys=False)
        print('extend data(ng): '+str(len(a_ext_data_ng)))

    print('** done **')
