#! python
# -*- coding: utf-8 -*-

import json
import random
import argparse

def count_non_valued_attribute(mr_value):
    nea = 0
    for attr in mr_value:
        if mr_value[attr] != '':
            nea += 1
    return nea

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_train', help='training data json file', default='../corpus/e2e_train.json')
    parser.add_argument('-nv', help='number of random variation', type=int, default=16)
    parser.add_argument('-d_data', help='data directory')
    parser.add_argument('-d_out', help='output directory')
    parser.add_argument('-n', help='n times of maxmum data size', type=int, default=1)
    args = parser.parse_args()

    print('** merge augmented data **')
    print(' original corpus data')
    print('  training               : '+str(args.f_train)) 
    print(' num of random variation : '+str(args.nv))
    print(' data directory          : '+str(args.d_data))
    print(' output directory        : '+str(args.d_out))
    print(' n                       : '+str(args.n))

    with open(args.f_train, 'r', encoding='utf-8') as f:
        a_train_org = json.load(f)

    a_nea = [0 for i in range(9)]
    with open(args.f_train, 'r', encoding='utf-8') as f:
        a_train_org = json.load(f)
    for obj in a_train_org:
        nea = count_non_valued_attribute(obj['mr']['value'])
        a_nea[nea] += 1
    max_nea = max(a_nea)
    num_aug = max_nea * args.n
    print(a_nea)
    print(str(max_nea))
    print(str(num_aug))

    a_train_aug = a_train_org
    for nea in range(1, 9):
        with open(args.d_data.rstrip('/')+'/ok_'+str(args.nv)+'_'+str(nea)+'.json', 'r', encoding='utf-8') as f:
            a_train_tmp = json.load(f)
        if len(a_train_tmp) < num_aug - a_nea[nea]:
            a_train_aug += a_train_tmp
        else:
            a_train_aug += a_train_tmp[:num_aug-a_nea[nea]]

    for obj in a_train_aug:
        if 'startword' in obj['mr']:
            del obj['mr']['startword']
        if ('num_sen' in obj['mr']) is False:
            obj['mr']['num_sen'] = obj['txt_lex'].count('.') + obj['txt_lex'].count('?')

    with open(args.d_out.rstrip('/')+'/e2e_train_aug_'+str(args.nv)+'_'+str(args.n)+'.json', 'w', encoding='utf-8') as f:
        json.dump(a_train_aug, f, ensure_ascii=False, indent=4, sort_keys=False)

    print('** done **')
