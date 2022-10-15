#! python

import argparse
import random
import json
import copy
import itertools

def conv_obj2string(obj):
    string = ''
    for i, attr in enumerate(obj):
        if i > 0:
            string += '|'
        string += str(obj[attr])
    return string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-np', help='number of order pattern for each data', type=int, default=4)
    parser.add_argument('-nv', help='number of random variation', type=int, default=4)
    parser.add_argument('-seed', help='random seed', type=int, default=1234)
    parser.add_argument('-d_i', help='original dataset directory', default='../../e2e_refined_dataset')
    parser.add_argument('-f_aug', help='augmented training data', default='../train/out/e2e_train_aug_16.json')
    parser.add_argument('-d_o', help='output directory', default='out')
    args = parser.parse_args()

    print('** generate augmented test data by shuffling MR order **')
    print(' original dataset directory         : '+str(args.d_i))
    print(' augmented training data file       : '+str(args.f_aug))
    print(' output directory                   : '+str(args.d_o))
    print(' num of order pattern for each data : '+str(args.np))
    print(' num of random variation            : '+str(args.nv))
    print(' random seed                        : '+str(args.seed))

    # existing dataset
    exist_mr_org = {}
    a_attribute = ['train', 'test']
    for attribute in a_attribute:
        with open(args.d_i.rstrip('/')+'/e2e_'+attribute+'.json', 'r', encoding='utf-8') as f:
            a_obj = json.load(f)
        for obj in a_obj:
            mr_string = conv_obj2string(obj['mr']['value_lex']) + '|' + conv_obj2string(obj['mr']['order'])
            if (mr_string in exist_mr_org) is False:
                exist_mr_org[mr_string] = ''
    with open(args.f_aug, 'r', encoding='utf-8') as f:
        a_obj = json.load(f)
    for obj in a_obj:
        mr_string = conv_obj2string(obj['mr']['value_lex']) + '|' + conv_obj2string(obj['mr']['order'])
        if (mr_string in exist_mr_org) is False:
            exist_mr_org[mr_string] = ''

    # generate all 'order' patterns
    a_order_pattern = {}
    for nea in range(1, 9):
        a_order_pattern[nea] = {'num': 0, 'order': []}
        a_num = []
        for i in range(nea):
            a_num.append(i+1)

        a_pattern = itertools.permutations(a_num)
        for pattern in a_pattern:
            a_tmp = []
            for m in range(nea):
                a_tmp.append(pattern[m])
            a_order_pattern[nea]['order'].append(a_tmp)
        a_order_pattern[nea]['num'] = len(a_order_pattern[nea]['order'])
    with open('order_pattern.json', 'w', encoding='utf-8') as f:
        json.dump(a_order_pattern, f, ensure_ascii=False, indent=4, sort_keys=False)

    # generate augmented data (w/ random order)
    with open(args.d_i.rstrip('/')+'/e2e_test.json', 'r', encoding='utf-8') as f:
        a_obj_org = json.load(f)

    a_nea = []
    a_pattern_idx = []
    for obj in a_obj_org:
        order = []
        nea = 0
        for attr in obj['mr']['order']:
            if obj['mr']['order'][attr] > 0:
                order.append(obj['mr']['order'][attr])
                nea += 1
        a_nea.append(nea)

        idx = 0
        for n in range(a_order_pattern[nea]['num']):
            if order == a_order_pattern[nea]['order'][n]:
                idx = n
                break
        a_pattern_idx.append(idx)

    for n in range(args.nv):
        # set random seed
        random.seed(args.seed + n)
        exist_mr_new = {}
        a_obj_new = []
        for i, obj in enumerate(a_obj_org):
            nea = a_nea[i]
            pattern_idx = a_pattern_idx[i]
            a_idx = random.sample(range(a_order_pattern[nea]['num']), k=a_order_pattern[nea]['num'])
            np = 0
            #print(str(obj['id'])+' nea: '+str(nea)+', pattern_idx: '+str(pattern_idx)+', len(a_idx): '+str(len(a_idx)))
            for j in range(len(a_idx)):
                if np >= args.np:
                    break
                idx = a_idx[j]
                if idx == pattern_idx:
                    continue

                obj_new = {
                    'id': obj['id'],
                    'id_ext': None,
                    'mr': {
                        'value': obj['mr']['value'],
                        'value_lex': obj['mr']['value_lex'],
                        'order': copy.deepcopy(obj['mr']['order'])
                    }
                }
                k = 0
                for attr in obj['mr']['order']:
                    if obj['mr']['order'][attr] != 0:
                        obj_new['mr']['order'][attr] = a_order_pattern[nea]['order'][idx][k]
                        k += 1

                # redundancy check
                mr_string = conv_obj2string(obj_new['mr']['value_lex']) + '|' + conv_obj2string(obj_new['mr']['order'])
                if mr_string in exist_mr_org:
                    #print(str(obj['id'])+': exist org')
                    continue
                if mr_string in exist_mr_new:
                    #print(str(obj['id'])+': exist new')
                    continue
                else:
                    #print(str(obj['id'])+': augmentation')
                    exist_mr_new[mr_string] = ''

                obj_new['id_ext'] = np
                np += 1
                a_obj_new.append(obj_new)

        with open(args.d_o.rstrip('\n')+'/e2e_test_aug_'+str(n)+'.json', 'w', encoding='utf-8') as f:
            json.dump(a_obj_new, f, ensure_ascii=False, indent=4, sort_keys=False)
            print(str(n)+'\t'+str(len(a_obj_new)))

    print('** done **')
