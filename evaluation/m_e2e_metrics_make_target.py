#! python
# -*- coding: utf-8 -*-

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input inference tsv file')
    parser.add_argument('-d_o', help='output directory')
    args = parser.parse_args()

    ## convert for e2e-metrics
    print('** convert for e2e-metrics target **')
    print(' input inference file : '+str(args.i))
    print(' output directory     : '+str(args.d_o))

    with open(args.i, 'r', encoding='utf-8') as f:
        a_inference = f.readlines()

    idx_value = 1
    idx_order = 2
    idx_txt = 4

    fo = open(args.d_o.rstrip('/')+'/target.txt', 'w', encoding='utf-8')
    for i in range(1, len(a_inference)):
        data = a_inference[i].rstrip('\n').split('\t')
        txt_predict = data[idx_txt]
        fo.write(txt_predict+'\n')
    fo.close()

    fo = open(args.d_o.rstrip('/')+'/target_value_merge.txt', 'w', encoding='utf-8')
    for i in range(1, len(a_inference)):
        data_i = a_inference[i].rstrip('\n').split('\t')
        value_i = data_i[idx_value]
        txt_predict = data_i[idx_txt]
        a_value = []
        for j in range(1, len(a_inference)):
            data_j = a_inference[j].rstrip('\n').split('\t')
            value_j = data_j[idx_value]
            if value_i == value_j:
                a_value.append(j)
        if (a_value[0] == i):
            fo.write(txt_predict+'\n')
    fo.close()

    fo = open(args.d_o.rstrip('/')+'/target_order_merge.txt', 'w', encoding='utf-8')
    for i in range(1, len(a_inference)):
        data_i = a_inference[i].rstrip('\n').split('\t')
        value_i = data_i[idx_value]
        order_i = data_i[idx_order]
        txt_predict = data_i[idx_txt]
        a_order = []
        for j in range(1, len(a_inference)):
            data_j = a_inference[j].rstrip('\n').split('\t')
            value_j = data_j[idx_value]
            order_j = data_j[idx_order]
            if (value_i == value_j) and (order_i == order_j):
                a_order.append(j)
        if (a_order[0] == i):
            fo.write(txt_predict+'\n')
    fo.close()

    print('** done **')
