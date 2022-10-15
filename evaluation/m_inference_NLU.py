#! python
# -*- coding: utf-8 -*-

import sys
import json
import argparse
sys.path.append('..')
from model.nlu import NLU

def conv_obj2string(obj):
    string = ''
    for i, attr in enumerate(obj):
        if i > 0:
            string += '|'
        string += str(obj[attr])
    return string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_model', help='model directory', default='../output/NLU')
    parser.add_argument('-f_model', help='model file', default='best.dat')
    parser.add_argument('-i', help='evaluation test data', default='../corpus/e2e_test.json')
    parser.add_argument('-o', help='output file', default='inference.tsv')
    parser.add_argument('-alg', help='O: value, A: value+order', default='A')
    args = parser.parse_args()

    print('** NLU evaluation **')
    print(' input data  : '+str(args.i))
    print(' output data : '+str(args.o))
    print(' model')
    print('  directory  : '+str(args.d_model))
    print('  file       : '+str(args.f_model))
    print(' algorithm   : '+str(args.alg))

    ## inference
    NLU_model = NLU(args.d_model, args.f_model, args.alg)

    with open(args.i, 'r', encoding='utf-8') as f_test:
        a_obj_in = json.load(f_test)

    fo = open(args.o, 'w', encoding='utf-8')
    if args.alg == 'A':
        fo.write('id\ttxt_lex\tmr_value(correct)\tmr_value(predict)\tresult(value)\t')
        fo.write('mr_order(correct)\tmr_order(predict)\tresult(order)\t')
        fo.write('result\n')

        for obj in a_obj_in:
            flag = True
            fo.write(str(obj['id'])+'\t'+obj['txt_lex']+'\t')
            out_mr_obj, _ = NLU_model.convert_nlu(obj['txt_lex'])

            # mr_value
            mr_value_string = conv_obj2string(obj['mr']['value_lex'])
            out_mr_value_string = conv_obj2string(out_mr_obj['value_lex'])
            fo.write(mr_value_string+'\t'+out_mr_value_string+'\t')
            if mr_value_string == out_mr_value_string:
                fo.write('True\t')
            else:
                flag = False
                fo.write('False\t')

            # mr_order
            mr_order_string = conv_obj2string(obj['mr']['order'])
            out_order_string = conv_obj2string(out_mr_obj['order'])
            fo.write(mr_order_string+'\t'+out_order_string+'\t')
            if mr_order_string == out_order_string:
                fo.write('True\t')
            else:
                flag = False
                fo.write('False\t')
            fo.write(str(flag)+'\n')

    else:
        # alg == 'O'
        fo.write('id\ttxt_lex\tmr_value(correct)\tmr_value(predict)\tresult\n')
        for obj in a_obj_in:
            fo.write(str(obj['id'])+'\t'+obj['txt_lex']+'\t')
            out_mr_obj, _ = NLU_model.convert_nlu(obj['txt_lex'])

            # mr_value
            mr_value_string = conv_obj2string(obj['mr']['value_lex'])
            out_mr_value_string = conv_obj2string(out_mr_obj['value_lex'])
            fo.write(mr_value_string+'\t'+out_mr_value_string+'\t')
            if mr_value_string == out_mr_value_string:
                fo.write('True\n')
            else:
                fo.write('False\n')
    fo.close()

    print('** done **')
