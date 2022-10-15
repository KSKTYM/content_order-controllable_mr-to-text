#! python
# -*- coding: utf-8 -*-

import sys
import json
import copy
import argparse
sys.path.append('..')
from model.nlu import NLU

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


def conv_obj2string(obj):
    string = ''
    for i, attr in enumerate(obj):
        if i > 0:
            string += '|'
        string += str(obj[attr])
    return string


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
    parser.add_argument('-d_model_nlu', help='parameter directory\'s name', default='../parameter')
    parser.add_argument('-f_model_nlu', help='network model file', default='best.dat')
    parser.add_argument('-i', help='input file', default='inference_test.tsv')
    parser.add_argument('-o', help='output file', default='inference_test_MRcheck.tsv')
    parser.add_argument('-alg_nlg', help='O: value, A: value+order', default='A')
    parser.add_argument('-alg_nlu', help='O: value, A: value+order', default='A')
    args = parser.parse_args()

    print('** NLG evaluation by NLU **')
    print(' input           : '+str(args.i))
    print(' output          : '+str(args.o))
    print('(NLU model)')
    print(' directory       : '+str(args.d_model_nlu))
    print(' model file      : '+str(args.f_model_nlu))
    print(' algorithm')
    print('  NLG            : '+str(args.alg_nlg))
    print('  NLU            : '+str(args.alg_nlu))

    NLU_model = NLU(args.d_model_nlu, args.f_model_nlu, args.alg_nlu)

    # inferenced data
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
    if args.alg_nlg == 'A':
        fo.write('id\t')
        fo.write('mr_value(correct)\ttxt(predict)\tmr_value(predict)\t')
        fo.write('name\teatType\tfood\tpriceRange\tcustomer rating\tarea\tfamilyFriendly\tnear\tresult(mr_value)\t')
        fo.write('mr_order(correct)\tmr_order(predict)\t')
        fo.write('order0\torder1\torder2\torder3\torder4\torder5\torder6\torder7\tresult(mr_order)\t')
        fo.write('result(all)\n')

        for i in range(1, len(a_input)):
            flag_error = {'all': False,
                          'value': False, 'value_del': False, 'value_ins': False, 'value_sub': False,
                          'order': False, 'order_del': False, 'order_ins': False, 'order_sub': False
            }

            idx = a_input[i].rstrip('\n').split('\t')[0]
            mr_value_correct_string = a_input[i].rstrip('\n').split('\t')[1]
            mr_order_correct_string = a_input[i].rstrip('\n').split('\t')[2]
            #txt_correct = a_input[i].rstrip('\n').split('\t')[3]
            txt_predict = a_input[i].rstrip('\n').split('\t')[4]

            fo.write(str(idx)+'\t')
            fo.write(mr_value_correct_string+'\t')
            fo.write(txt_predict+'\t')

            # mr_value
            mr_value_correct, nea = conv_string2obj(mr_value_correct_string)
            mr_value_lex_correct = copy.deepcopy(mr_value_correct)
            txt_lex_predict = txt_predict
            if mr_value_correct['name'] != '':
                mr_value_lex_correct['name'] = 'NAME'
                txt_lex_predict = txt_lex_predict.replace(mr_value_correct['name'], 'NAME')
            if mr_value_correct['near'] != '':
                mr_value_lex_correct['near'] = 'NEAR'
                txt_lex_predict = txt_lex_predict.replace(mr_value_correct['near'], 'NEAR')

            mr_predict, _ = NLU_model.convert_nlu(txt_lex_predict)
            mr_value_predict_string = conv_obj2string(mr_predict['value_lex'])
            fo.write(mr_value_predict_string+'\t')

            # count(mr_value)
            a_count[nea-1] += 1
            for attr in mr_value_lex_correct:
                if str(mr_value_lex_correct[attr]) == str(mr_predict['value_lex'][attr]):
                    fo.write('OK\t')
                else:
                    flag_error['all'] = True
                    flag_error['value'] = True
                    if (str(mr_value_lex_correct[attr]) != '') and (str(mr_predict['value_lex'][attr]) == ''):
                        fo.write('del\t')
                        flag_error['value_del'] = True
                    elif (str(mr_value_lex_correct[attr]) == '') and (str(mr_predict['value_lex'][attr]) != ''):
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
            mr_order_correct, _ = conv_string2obj(mr_order_correct_string)
            mr_order_predict_string = conv_obj2string(mr_predict['order'])
            fo.write(mr_order_correct_string+'\t')
            fo.write(mr_order_predict_string+'\t')

            # count(mr_order)
            for attr in mr_order_correct:
                if str(mr_order_correct[attr]) == str(mr_predict['order'][attr]):
                    fo.write('OK\t')
                else:
                    flag_error['all'] = True
                    flag_error['order'] = True
                    if (str(mr_order_correct[attr]) != '') and (str(mr_predict['order'][attr]) == ''):
                        fo.write('del\t')
                        flag_error['order_del'] = True
                    elif (str(mr_order_correct[attr]) == '') and (str(mr_predict['order'][attr]) != ''):
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
        fo.close()

    else:
        fo.write('id\t')
        fo.write('mr_value(correct)\ttxt(predict)\tmr_value(predict)\t')
        fo.write('name\teatType\tfood\tpriceRange\tcustomer rating\tarea\tfamilyFriendly\tnear\tresult(mr_value)\n')

        for i in range(1, len(a_input)):
            flag_error = {'all': False,
                          'value': False, 'value_del': False, 'value_ins': False, 'value_sub': False
            }

            idx = a_input[i].rstrip('\n').split('\t')[0]
            mr_value_correct_string = a_input[i].rstrip('\n').split('\t')[1]
            #mr_order_correct_string = a_input[i].rstrip('\n').split('\t')[2]
            #txt_correct = a_input[i].rstrip('\n').split('\t')[3]
            txt_predict = a_input[i].rstrip('\n').split('\t')[4]

            fo.write(str(idx)+'\t')
            fo.write(mr_value_correct_string+'\t')
            fo.write(txt_predict+'\t')

            # mr_value
            mr_value_correct, nea = conv_string2obj(mr_value_correct_string)
            mr_value_lex_correct = copy.deepcopy(mr_value_correct)
            txt_lex_predict = txt_predict
            if mr_value_correct['name'] != '':
                mr_value_lex_correct['name'] = 'NAME'
                txt_lex_predict = txt_lex_predict.replace(mr_value_correct['name'], 'NAME')
            if mr_value_correct['near'] != '':
                mr_value_lex_correct['near'] = 'NEAR'
                txt_lex_predict = txt_lex_predict.replace(mr_value_correct['near'], 'NEAR')

            mr_predict, _ = NLU_model.convert_nlu(txt_lex_predict)
            mr_value_predict_string = conv_obj2string(mr_predict['value_lex'])
            fo.write(mr_value_predict_string+'\t')

            # count(mr_value)
            a_count[nea-1] += 1
            for attr in mr_value_lex_correct:
                if str(mr_value_lex_correct[attr]) == str(mr_predict['value_lex'][attr]):
                    fo.write('OK\t')
                else:
                    flag_error['all'] = True
                    flag_error['value'] = True
                    if (str(mr_value_lex_correct[attr]) != '') and (str(mr_predict['value_lex'][attr]) == ''):
                        fo.write('del\t')
                        flag_error['value_del'] = True
                    elif (str(mr_value_lex_correct[attr]) == '') and (str(mr_predict['value_lex'][attr]) != ''):
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
    if args.alg_nlg == 'A':
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
