#! python
# -*- coding: utf-8 -*-

import json
import argparse

def conv_obj2string(obj):
    string = ''
    for i, attr in enumerate(obj):
        if i > 0:
            string += '|'
        string += str(obj[attr])
    return string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', help='test corpus json file')
    parser.add_argument('-t', help='TGEN inference txt file')
    parser.add_argument('-o', help='output tsv file')
    args = parser.parse_args()

    print('** convert tgen data **')
    print(' test corpus json file   : '+str(args.c))
    print(' TGEN inference txt file : '+str(args.t))
    print(' output tsv file         : '+str(args.o))

    with open(args.c, 'r', encoding='utf-8') as f:
        a_obj = json.load(f)
    with open(args.t, 'r', encoding='utf-8') as f:
        a_inference = f.readlines()
    fo = open(args.o, 'w', encoding='utf-8')

    fo.write('id\tmr_value\tmr_order\ttxt(correct)\ttxt(predict)\tresult\n')

    for i, obj in enumerate(a_obj):
        idx = obj['id']
        mr_value_string = conv_obj2string(obj['mr']['value'])
        mr_order_string = conv_obj2string(obj['mr']['order'])
        txt_correct = obj['txt']
        txt_predict = a_inference[i].rstrip('\n')

        mr_food = obj['mr']['value']['food']
        if (mr_food != '') and \
           (mr_food.lower() in txt_predict):
            txt_predict = txt_predict.replace(mr_food.lower(), mr_food)

        if txt_correct == txt_predict:
            result = True
        else:
            result = False
        fo.write(str(idx)+'\t')
        fo.write(str(mr_value_string)+'\t')
        fo.write(str(mr_order_string)+'\t')
        fo.write(str(txt_correct)+'\t')
        fo.write(str(txt_predict)+'\t')
        fo.write(str(result)+'\n')
    fo.close()
    print('** done **')
