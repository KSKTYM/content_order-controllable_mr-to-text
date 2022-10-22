#! python
# -*- coding: utf-8 -*-

import sys
import json
import argparse
sys.path.append('..')
from model.nlg import NLG

def conv_obj2string(obj):
    string = ''
    for i, attr in enumerate(obj):
        if i > 0:
            string += '|'
        string += str(obj[attr])
    return string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_model_nlg', help='model directory (NLG)', default='../output/NLG')
    parser.add_argument('-f_model_nlg', help='model file (NLG)', default='best.dat')
    parser.add_argument('-d_model_nlu', help='model directory (NLU)', default='../output/NLU')
    parser.add_argument('-f_model_nlu', help='model file (NLU)', default='best.dat')
    parser.add_argument('-search', help='search method([best]|beam|greedy)', choices=['best', 'beam', 'greedy'], default='best')
    parser.add_argument('-i', help='evaluation test data', default='../corpus/e2e_test.json')
    parser.add_argument('-aug', help='augmented test data', action='store_true')
    parser.add_argument('-o', help='output file', default='result.tsv')
    parser.add_argument('-alg_nlg', help='O: value, A: value+order', default='A')
    parser.add_argument('-alg_nlu', help='O: value, A: value+order', default='A')
    args = parser.parse_args()

    print('** NLG evaluation **')
    print(' input data      : '+str(args.i))
    print('  augmented data : '+str(args.aug))
    print(' output data     : '+str(args.o))
    print(' model(NLG)')
    print('  directory      : '+str(args.d_model_nlg))
    print('  file           : '+str(args.f_model_nlg))
    print(' model(NLU)')
    print('  directory      : '+str(args.d_model_nlu))
    print('  file           : '+str(args.f_model_nlu))
    print(' decoder')
    print('  search method  : '+str(args.search))
    print(' algorithm')
    print('  NLG            : '+str(args.alg_nlg))
    print('  NLU            : '+str(args.alg_nlu))

    NLG_model = NLG(args.d_model_nlg, args.f_model_nlg,
                    args.d_model_nlu, args.f_model_nlu,
                    args.alg_nlg, args.alg_nlu)

    with open(args.i, 'r', encoding='utf-8') as f_test:
        a_obj_in = json.load(f_test)

    prev = {
        'mr_value': None,
        'mr_order': None,
        'output_text': None
    }

    fo = open(args.o, 'w', encoding='utf-8')
    fo.write('id\tmr_value\tmr_order\ttxt(correct)\ttxt(predict)\tresult\n')
    for obj in a_obj_in:
        mr_value_string = conv_obj2string(obj['mr']['value'])
        mr_order_string = conv_obj2string(obj['mr']['order'])
        if args.aug is True:
            txt = ''
        else:
            txt = obj['txt']

        if (mr_value_string != prev['mr_value']) or \
           (mr_order_string != prev['mr_order']):
            output_text, _ = NLG_model.convert_nlg(args.search, obj['mr'])
        else:
            output_text = prev['output_text']
        if 'id_ext' in obj:
            fo.write(str(obj['id'])+'-'+str(obj['id_ext'])+'\t')
        else:
            fo.write(str(obj['id'])+'\t')
        fo.write(mr_value_string+'\t')
        fo.write(mr_order_string+'\t')
        fo.write(txt+'\t'+output_text+'\t')

        if txt == output_text:
            fo.write('True\n')
        else:
            fo.write('False\n')

        prev['mr_value'] = mr_value_string
        prev['mr_order'] = mr_order_string
        prev['output_text'] = output_text
    fo.close()
    print('** done **')
