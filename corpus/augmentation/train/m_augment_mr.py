#! python
# -*- coding: utf-8 -*-

import json
import random
import itertools
import argparse

def conv_value2string(mr):
    string = mr['name']+'|'
    string += mr['eatType']+'|'
    string += mr['food']+'|'
    string += mr['priceRange']+'|'
    string += mr['customer rating']+'|'
    string += mr['area']+'|'
    string += mr['familyFriendly']+'|'
    string += mr['near']
    return string

def conv_order2string(order):
    string = str(order['name'])+'|'
    string += str(order['eatType'])+'|'
    string += str(order['food'])+'|'
    string += str(order['priceRange'])+'|'
    string += str(order['customer rating'])+'|'
    string += str(order['area'])+'|'
    string += str(order['familyFriendly'])+'|'
    string += str(order['near'])
    return string

def make_mr_obj(mr_value, mr_order):
    mr_obj = {
        'value_lex': mr_value,
        'order': {
            "name": 0,
            "eatType": 0,
            "food": 0,
            "priceRange": 0,
            "customer rating": 0,
            "area": 0,
            "familyFriendly": 0,
            "near": 0
        }
    }
    i = 0
    for attr in mr_value:
        if mr_value[attr] != '':
            mr_obj['order'][attr] = mr_order[i]
            i += 1
    return mr_obj

def count_non_valued_attribute(mr_value):
    nea = 0
    for attr in mr_value:
        if mr_value[attr] != '':
            nea += 1
    return nea

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_i', help='original dataset directory', default='../../e2e_refined_dataset')
    parser.add_argument('-d_data', help='output directory')
    parser.add_argument('-seed', help='random seed', type=int, default=1234)
    parser.add_argument('-nv', help='number of random variation', type=int, default=16)
    args = parser.parse_args()

    print('** generate augmented MR data **')
    print(' original dataset directory : '+str(args.d_i))
    print(' data directory             : '+str(args.d_data))
    print(' random seed                : '+str(args.seed))
    print(' num of random variation    : '+str(args.nv))

    ##
    ## generate augmented MR data
    ##
    random.seed(args.seed)

    # collect MR values
    mr_value_list = {
        'name': [],
        'eatType': [],
        'food': [],
        'priceRange': [],
        'customer rating': [],
        'area': [],
        'familyFriendly': [],
        'near': []
    }

    # existing MR pattern
    exist_mr = {}
    a_attribute = ['train', 'valid', 'test']
    for attribute in a_attribute:
        with open(args.d_i.rstrip('/')+'/e2e_'+attribute+'.json', 'r', encoding='utf-8') as f:
            a_obj = json.load(f)
        for obj in a_obj:
            for attr in mr_value_list:
                if (obj['mr']['value_lex'][attr] in mr_value_list[attr]) is False:
                    mr_value_list[attr].append(obj['mr']['value_lex'][attr])
            mr_value_string = conv_value2string(obj['mr']['value_lex'])
            mr_order_string = conv_order2string(obj['mr']['order'])
            mr_string = mr_value_string + '|' + mr_order_string
            if (mr_string in exist_mr) is False:
                exist_mr[mr_string] = ''
    with open(args.d_data.rstrip('/')+'/mr_value_list.json', 'w', encoding='utf-8') as f:
        json.dump(mr_value_list, f, ensure_ascii=False, indent=4, sort_keys=False)
    with open(args.d_data.rstrip('/')+'/exist_mr.json', 'w', encoding='utf-8') as f:
        json.dump(exist_mr, f, ensure_ascii=False, indent=4, sort_keys=False)

    # MRorder pattern (all combination)
    a_order_pattern = {}
    for nea in range(1, 9):
        a_order_pattern[nea] = {'pattern': {'order': 0, 'value': 0}, 'order': []}
        a_num = []
        for i in range(nea):
            a_num.append(i+1)

        a_pattern = itertools.permutations(a_num)
        for pattern in a_pattern:
            a_tmp = []
            for m in range(nea):
                a_tmp.append(pattern[m])
            a_order_pattern[nea]['order'].append(a_tmp)
        a_order_pattern[nea]['pattern']['order'] = len(a_order_pattern[nea]['order'])

    # generate MR combination
    a_out = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    for value_A in mr_value_list['eatType']:
        for value_B in mr_value_list['food']:
            for value_C in mr_value_list['priceRange']:
                for value_D in mr_value_list['customer rating']:
                    for value_E in mr_value_list['area']:
                        for value_F in mr_value_list['familyFriendly']:
                            for value_G in mr_value_list['near']:
                                mr_value = {
                                    'name': 'NAME',
                                    'eatType': value_A,
                                    'food': value_B,
                                    'priceRange': value_C,
                                    'customer rating': value_D,
                                    'area': value_E,
                                    'familyFriendly': value_F,
                                    'near': value_G
                                }
                                mr_value_string = conv_value2string(mr_value)
                                nea = count_non_valued_attribute(mr_value)
                                nv = min(a_order_pattern[nea]['pattern']['order'], args.nv)
                                a_idx = random.sample(range(a_order_pattern[nea]['pattern']['order']), k = nv)
                                for j in range(len(a_idx)):
                                    mr_obj = make_mr_obj(mr_value, a_order_pattern[nea]['order'][a_idx[j]])
                                    mr_string = mr_value_string + '|' + conv_order2string(mr_obj['order'])
                                    if (mr_string in exist_mr) is False:
                                        a_out[nea].append(mr_obj)
                                        a_order_pattern[nea]['pattern']['value'] += 1
    del exist_mr

    with open(args.d_data.rstrip('/')+'/mr_combination_'+str(args.nv)+'.json', 'w', encoding='utf-8') as f:
        json.dump(a_out, f, ensure_ascii=False, indent=4, sort_keys=False)
        num_data = 0
        for nea in a_out:
            num_data += len(a_out[nea])
        print('e2e_train_mr_aug_'+str(args.nv)+'.json has '+str(num_data)+' data')
    del a_out
    with open(args.d_data.rstrip('/')+'/mr_order_pattern_'+str(args.nv)+'.json', 'w', encoding='utf-8') as f:
        json.dump(a_order_pattern, f, ensure_ascii=False, indent=4, sort_keys=False)

    all_pattern = 0
    all_num = 0
    for nea in a_order_pattern:
        all_pattern += a_order_pattern[nea]['pattern']['order']
        all_num += a_order_pattern[nea]['pattern']['value']
        print(str(nea)+'\t'+str(a_order_pattern[nea]['pattern']['order'])+'\t'+str(a_order_pattern[nea]['pattern']['value']))
    print('all pattern: '+str(all_pattern))
    print('all num: '+str(all_num))
    print('** done(MR) **')
