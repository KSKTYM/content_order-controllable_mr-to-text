#! python
# -*- coding: utf-8 -*-

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input file', default='../corpus/e2e_refined_dataset/e2e_test.json')
    parser.add_argument('-d_o', help='output directory', default='e2e-metrics-data')
    args = parser.parse_args()

    ## convert for e2e-metrics
    print('** convert for e2e-metrics **')
    print(' input file       : '+str(args.i))
    print(' output directory : '+str(args.d_o))

    with open(args.i, 'r', encoding='utf-8') as f:
        a_obj = json.load(f)

    fo_v = open(args.d_o.rstrip('/')+'/reference_value.txt', 'w', encoding='utf-8')
    fo_o = open(args.d_o.rstrip('/')+'/reference_order.txt', 'w', encoding='utf-8')

    count_iv = 0
    count_io = 0
    for i, obj_i in enumerate(a_obj):
        if i > 0:
            fo_v.write('\n')
            fo_o.write('\n')
        count_jv = 0
        count_jo = 0
        for j, obj_j in enumerate(a_obj):
            if obj_i['mr']['value'] == obj_j['mr']['value']:
                fo_v.write(obj_j['txt']+'\n')
                count_jv += 1
                if obj_i['mr']['order'] == obj_j['mr']['order']:
                    fo_o.write(obj_j['txt']+'\n')
                    count_jo += 1
        count_iv += count_jv
        count_io += count_jo
    print('ave of num pairs (value)          : '+str(count_iv / len(a_obj)))
    print('ave of num pairs (value/order)    : '+str(count_io / len(a_obj)))
    fo_v.close()
    fo_o.close()

    fo_v = open(args.d_o.rstrip('/')+'/reference_value_merge.txt', 'w', encoding='utf-8')
    fo_o = open(args.d_o.rstrip('/')+'/reference_order_merge.txt', 'w', encoding='utf-8')

    flag_v = False
    flag_o = False
    count_v = 0
    count_o = 0
    for i, obj_i in enumerate(a_obj):
        a_value = []
        a_order = []
        for j, obj_j in enumerate(a_obj):
            if obj_i['mr']['value'] == obj_j['mr']['value']:
                a_value.append(j)
                if obj_i['mr']['order'] == obj_j['mr']['order']:
                    a_order.append(j)

        if (a_value[0] == i):
            if flag_v is True:
                fo_v.write('\n')
            else:
                flag_v = True
            count_v += 1
            for k in range(len(a_value)):
                fo_v.write(a_obj[a_value[k]]['txt']+'\n')

        if (a_order[0] == i):
            if flag_o is True:
                fo_o.write('\n')
            else:
                flag_o = True
            count_o += 1
            for k in range(len(a_order)):
                fo_o.write(a_obj[a_order[k]]['txt']+'\n')

    print('count_value: '+str(count_v))
    print('count_order: '+str(count_o))

    fo_v.close()
    fo_o.close()
    print('** done **')
