#! python
# -*- coding: utf-8 -*-

import sys
import json
import copy
import argparse

def conv_string2obj(text):
    mr = {}
    mr['name'] = text.split('|')[0]
    mr['eatType'] = text.split('|')[1]
    mr['food'] = text.split('|')[2]
    mr['priceRange'] = text.split('|')[3]
    mr['customer rating'] = text.split('|')[4]
    mr['area'] = text.split('|')[5]
    mr['familyFriendly'] = text.split('|')[6]
    mr['near'] = text.split('|')[7]
    nea = 0
    for attr in mr:
        if mr[attr] != '':
            nea += 1
    return mr, nea


def func_result_success(fo, a_count, a_error, msg_string):
    fo.write('['+msg_string+']\n')
    print('['+msg_string+']')
    success_all = 0
    for nea in range(1, 9):
        if a_count[nea-1] == 0:
            continue
        fo.write('NEA:'+str(nea)+'\t'+str(100.0*(a_count[nea-1]-a_error[nea-1])/a_count[nea-1])+'[%]\t('+str((a_count[nea-1]-a_error[nea-1]))+'/'+str(a_count[nea-1])+')\n')
        print('NEA:'+str(nea)+'\t'+str(100.0*(a_count[nea-1]-a_error[nea-1])/a_count[nea-1])+'[%]\t('+str((a_count[nea-1]-a_error[nea-1]))+'/'+str(a_count[nea-1])+')')
        success_all += a_count[nea-1] - a_error[nea-1]
    fo.write('total\t'+str(100.0*success_all/sum(a_count))+'[%]\t('+str(success_all)+'/'+str(sum(a_count))+')\n')
    print('total\t'+str(100.0*success_all/sum(a_count))+'[%]\t('+str(success_all)+'/'+str(sum(a_count))+')')
    return


def func_result_error(fo, a_count, a_error, msg_string):
    fo.write('['+msg_string+']\n')
    print('['+msg_string+']')
    for nea in range(1, 9):
        if a_count[nea-1] == 0:
            continue
        fo.write('NEA:'+str(nea)+'\t'+str(100.0*a_error[nea-1]/a_count[nea-1])+'[%]\t('+str(a_error[nea-1])+'/'+str(a_count[nea-1])+')\n')
        print('NEA:'+str(nea)+'\t'+str(100.0*a_error[nea-1]/a_count[nea-1])+'[%]\t('+str(a_error[nea-1])+'/'+str(a_count[nea-1])+')')
    fo.write('total\t'+str(100.0*sum(a_error)/sum(a_count))+'[%]\t('+str(sum(a_error))+'/'+str(sum(a_count))+')\n')
    print('total\t'+str(100.0*sum(a_error)/sum(a_count))+'[%]\t('+str(sum(a_error))+'/'+str(sum(a_count))+')')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input file', default='inference_test.tsv')
    parser.add_argument('-o', help='output file', default='inference_test_MRcheck.tsv')
    parser.add_argument('-alg', help='O: value, A: value+order', default='A')
    args = parser.parse_args()

    print('** NLU evaluation **')
    print(' input     : '+str(args.i))
    print(' output    : '+str(args.o))
    print(' algorithm : '+str(args.alg))

    with open(args.i, 'r', encoding='utf-8') as fi:
        a_input = fi.readlines()

    a_all_error = [0 for i in range(8)]
    a_value_error = [0 for i in range(8)]
    a_value_error_del = [0 for i in range(8)]
    a_value_error_ins = [0 for i in range(8)]
    a_value_error_sub = [0 for i in range(8)]
    a_order_error = [0 for i in range(8)]
    a_order_error_del = [0 for i in range(8)]
    a_order_error_ins = [0 for i in range(8)]
    a_order_error_sub = [0 for i in range(8)]
    a_idx_sen_error = [0 for i in range(8)]
    a_idx_sen_error_del = [0 for i in range(8)]
    a_idx_sen_error_ins = [0 for i in range(8)]
    a_idx_sen_error_sub = [0 for i in range(8)]
    a_num_sen_error = [0 for i in range(8)]
    a_count = [0 for i in range(8)]

    fo = open(args.o, 'w', encoding='utf-8')
    fo.write('id\t')
    fo.write('txt_lex\tmr_lex(correct)\tmr_lex(predict)\t')
    if args.alg == 'A':
        fo.write('name\teatType\tfood\tpriceRange\tcustomer rating\tarea\tfamilyFriendly\tnear\tresult(value)\t')
        fo.write('order(correct)\torder(predict)\t')
        fo.write('order0\torder1\torder2\torder3\torder4\torder5\torder6\torder7\tresult(order)\t')
        fo.write('result(all)\n')

        for i in range(1, len(a_input)):
            flag_error = {'all': False,
                          'value': False, 'value_del': False, 'value_ins': False, 'value_sub': False,
                          'order': False, 'order_del': False, 'order_ins': False, 'order_sub': False
            }

            idx = a_input[i].rstrip('\n').split('\t')[0]
            txt_lex = a_input[i].rstrip('\n').split('\t')[1]
            mr_value_correct_string = a_input[i].rstrip('\n').split('\t')[2]
            mr_value_predict_string = a_input[i].rstrip('\n').split('\t')[3]
            mr_order_correct_string = a_input[i].rstrip('\n').split('\t')[5]
            mr_order_predict_string = a_input[i].rstrip('\n').split('\t')[6]

            fo.write(idx+'\t')
            fo.write(txt_lex+'\t')
            fo.write(mr_value_correct_string+'\t')
            fo.write(mr_value_predict_string+'\t')

            # mr_value
            mr_value_correct, nea = conv_string2obj(mr_value_correct_string)
            mr_value_predict, _ = conv_string2obj(mr_value_predict_string)

            # count(mr_value)
            a_count[nea-1] += 1
            for attr in mr_value_correct:
                if str(mr_value_correct[attr]) == str(mr_value_predict[attr]):
                    fo.write('OK\t')
                else:
                    flag_error['all'] = True
                    flag_error['value'] = True
                    if (str(mr_value_correct[attr]) != '') and (str(mr_value_predict[attr]) == ''):
                        fo.write('del\t')
                        flag_error['value_del'] = True
                    elif (str(mr_value_correct[attr]) == '') and (str(mr_value_predict[attr]) != ''):
                        fo.write('ins\t')
                        flag_error['value_ins'] = True
                    else:
                        fo.write('sub\t')
                        flag_error['value_sub'] = True

            if flag_error['value'] is True:
                a_value_error[nea-1] += 1
            if flag_error['value'] is True:
                fo.write('NG\t')
            else:
                fo.write('OK\t')
            if flag_error['value_del'] is True:
                a_value_error_del[nea-1] += 1
            if flag_error['value_ins'] is True:
                a_value_error_ins[nea-1] += 1
            if flag_error['value_sub'] is True:
                a_value_error_sub[nea-1] += 1

            # mr_order
            fo.write(mr_order_correct_string+'\t')
            fo.write(mr_order_predict_string+'\t')
            mr_order_correct, _ = conv_string2obj(mr_order_correct_string)
            mr_order_predict, _ = conv_string2obj(mr_order_predict_string)

            # count(mr_order)
            for attr in mr_order_correct:
                if str(mr_order_correct[attr]) == str(mr_order_predict[attr]):
                    fo.write('OK\t')
                else:
                    flag_error['all'] = True
                    flag_error['order'] = True
                    if (str(mr_order_correct[attr]) != '') and (str(mr_order_predict[attr]) == ''):
                        fo.write('del\t')
                        flag_error['order_del'] = True
                    elif (str(mr_order_correct[attr]) == '') and (str(mr_order_predict[attr]) != ''):
                        fo.write('ins\t')
                        flag_error['order_ins'] = True
                    else:
                        fo.write('sub\t')
                        flag_error['order_sub'] = True

            if flag_error['order'] is True:
                a_order_error[nea-1] += 1
                fo.write('NG\t')
            else:
                fo.write('OK\t')
            if flag_error['order_del'] is True:
                a_order_error_del[nea-1] += 1
            if flag_error['order_ins'] is True:
                a_order_error_ins[nea-1] += 1
            if flag_error['order_sub'] is True:
                a_order_error_sub[nea-1] += 1

            # all
            if flag_error['all'] is True:
                a_all_error[nea-1] += 1
                fo.write('NG\n')
            else:
                fo.write('OK\n')

    else:
        fo.write('name\teatType\tfood\tpriceRange\tcustomer rating\tarea\tfamilyFriendly\tnear\tresult\n')
        for i in range(1, len(a_input)):
            flag_error = {'all': False,
                          'value': False, 'value_del': False, 'value_ins': False, 'value_sub': False,
                          'order': False, 'order_del': False, 'order_ins': False, 'order_sub': False
            }

            idx = a_input[i].rstrip('\n').split('\t')[0]
            txt_lex = a_input[i].rstrip('\n').split('\t')[1]
            mr_value_correct_string = a_input[i].rstrip('\n').split('\t')[2]
            mr_value_predict_string = a_input[i].rstrip('\n').split('\t')[3]

            fo.write(idx+'\t')
            fo.write(txt_lex+'\t')
            fo.write(mr_value_correct_string+'\t')
            fo.write(mr_value_predict_string+'\t')

            # mr_value
            mr_value_correct, nea = conv_string2obj(mr_value_correct_string)
            mr_value_predict, _ = conv_string2obj(mr_value_predict_string)

            # count(mr_value)
            a_count[nea-1] += 1
            for attr in mr_value_correct:
                if str(mr_value_correct[attr]) == str(mr_value_predict[attr]):
                    fo.write('OK\t')
                else:
                    flag_error['all'] = True
                    flag_error['value'] = True
                    if (str(mr_value_correct[attr]) != '') and (str(mr_value_predict[attr]) == ''):
                        fo.write('del\t')
                        flag_error['value_del'] = True
                    elif (str(mr_value_correct[attr]) == '') and (str(mr_value_predict[attr]) != ''):
                        fo.write('ins\t')
                        flag_error['value_ins'] = True
                    else:
                        fo.write('sub\t')
                        flag_error['value_sub'] = True

            if flag_error['value'] is True:
                a_value_error[nea-1] += 1
            if flag_error['value'] is True:
                fo.write('NG\t')
            else:
                fo.write('OK\t')
            if flag_error['value_del'] is True:
                a_value_error_del[nea-1] += 1
            if flag_error['value_ins'] is True:
                a_value_error_ins[nea-1] += 1
            if flag_error['value_sub'] is True:
                a_value_error_sub[nea-1] += 1
            # all
            if flag_error['all'] is True:
                a_all_error[nea-1] += 1
                fo.write('NG\n')
            else:
                fo.write('OK\n')

    fo.close()

    fo = open(args.o[:-4]+'_score.txt', 'w', encoding='utf-8')
    if args.alg == 'A':
        # all accuracy
        func_result_success(fo, a_count, a_all_error, 'all accuracy')

        # mr_value accuracy
        func_result_success(fo, a_count, a_value_error, 'mr_value accuracy')

        # mr_order accuracy
        func_result_success(fo, a_count, a_order_error, 'mr_order accuracy')

        # all errors
        func_result_error(fo, a_count, a_all_error, 'all error')

        ## mr_value error
        func_result_error(fo, a_count, a_value_error, 'mr_value error')

        ## mr_value deletion error
        func_result_error(fo, a_count, a_value_error_del, 'mr_value deletion error')

        ## mr_value insertion error
        func_result_error(fo, a_count, a_value_error_ins, 'mr_value insertion error')

        ## mr_value substitution error
        func_result_error(fo, a_count, a_value_error_sub, 'mr_value substitution error')

        ## mr_order error
        func_result_error(fo, a_count, a_order_error, 'mr_order error')

        ## mr_order deletion error
        func_result_error(fo, a_count, a_order_error_del, 'mr_order deletion error')

        ## mr_order insertion error
        func_result_error(fo, a_count, a_order_error_ins, 'mr_order insertion error')

        ## mr_order substitution error
        func_result_error(fo, a_count, a_order_error_sub, 'mr_order substitution error')

    else:
        # all accuracy
        func_result_success(fo, a_count, a_all_error, 'all accuracy')

        # mr_value accuracy
        func_result_success(fo, a_count, a_value_error, 'mr_value accuracy')

        # mr_order accuracy
        func_result_success(fo, a_count, a_order_error, 'mr_order accuracy')

        # all errors
        func_result_error(fo, a_count, a_all_error, 'all error')

        ## mr_value error
        func_result_error(fo, a_count, a_value_error, 'mr_value error')

        ## mr_value deletion error
        func_result_error(fo, a_count, a_value_error_del, 'mr_value deletion error')

        ## mr_value insertion error
        func_result_error(fo, a_count, a_value_error_ins, 'mr_value insertion error')

        ## mr_value substitution error
        func_result_error(fo, a_count, a_value_error_sub, 'mr_value substitution error')

    fo.close()
    print('** done **')
