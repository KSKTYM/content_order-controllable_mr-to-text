#! python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim

import math
import datetime
import time

import train
from dataset import MyDataset_O, MyDataset_A
from dictionary import Dictionary
sys.path.append('..')
from tokenizer.tokenizer import Tokenizer
from model.model_nlg import NLG_Encoder, NLG_Decoder, NLG_Model

## sub functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


## main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_train', help='training data json file', default='../corpus/e2e_refined_dataset/e2e_train.json')
    parser.add_argument('-f_valid', help='validation data json file', default='../corpus/e2e_refined_dataset/e2e_valid.json')
    parser.add_argument('-f_test', help='test data json file', default='../corpus/e2e_refined_dataset/e2e_test.json')
    parser.add_argument('-d_model', help='output model directory', default='../output/NLG/')
    parser.add_argument('-epoch', help='epoch number', type=int, default=100)
    parser.add_argument('-batch', help='batch size', type=int, default=128)
    parser.add_argument('-lr', help='learning rate', type=float, default=5e-04)
    parser.add_argument('-dropout', help='dropout rate', type=float, default=0.1)
    parser.add_argument('-clip', help='clip parameter', type=float, default=1.0)
    parser.add_argument('-seed', help='random seed number', type=int, default=1234)
    parser.add_argument('-hid_dim', help='hidden vector dimension', type=int, default=256)
    parser.add_argument('-enc_layer', help='number of layers in encoder', type=int, default=3)
    parser.add_argument('-dec_layer', help='number of layers in decoder', type=int, default=3)
    parser.add_argument('-enc_head', help='number of heads in encoder', type=int, default=8)
    parser.add_argument('-dec_head', help='number of heads in decoder', type=int, default=8)
    parser.add_argument('-enc_pf_dim', help='pf vector dimension in encoder', type=int, default=512)
    parser.add_argument('-dec_pf_dim', help='pf vector dimension in decoder', type=int, default=512)
    parser.add_argument('-alg', help='O: value, A: value+order', default='A')
    parser.add_argument('-v', help='verbose(print debug)', action='store_true')
    args = parser.parse_args()

    print('** NLG training **')
    print(' corpus data')
    print('  training data        : '+str(args.f_train))
    print('  validation data      : '+str(args.f_valid))
    print('  test data            : '+str(args.f_test))
    print(' output directory      : '+str(args.d_model))
    print(' training parameters')
    print('  epoch number         : '+str(args.epoch))
    print('  batch size           : '+str(args.batch))
    print('  learning rate        : '+str(args.lr))
    print('  dropout rate         : '+str(args.dropout))
    print('  clip                 : '+str(args.clip))
    print('  random seed number   : '+str(args.seed))
    print(' transformer parameters')
    print('  hidden dimension     : '+str(args.hid_dim))
    print('  encoder')
    print('   num of layers       : '+str(args.enc_layer))
    print('   num of heads        : '+str(args.enc_head))
    print('   pf vector dimension : '+str(args.enc_pf_dim))
    print('  decoder')
    print('   num of layers       : '+str(args.dec_layer))
    print('   num of heads        : '+str(args.dec_head))
    print('   pf vector dimension : '+str(args.dec_pf_dim))
    print(' algorithm             : '+str(args.alg))
    if args.v is True:
        print(' verbose (print debug) : ON')

    # (1) torch settings
    print('(1) torch settings')
    print('  version              : '+str(torch.__version__))
    print('  cuda                 : '+str(torch.cuda.is_available()))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # (2) dataset
    print('(2) dataset')
    tokenizer = Tokenizer()
    dictionary = Dictionary(args.f_train, args.f_valid, args.f_test, tokenizer)
    e2e_dictionary = dictionary.get_dictionary()
    if not os.path.exists(args.d_model):
        os.mkdir(args.d_model)
    with open(args.d_model.rstrip('/')+'/dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(e2e_dictionary, f, ensure_ascii=False, indent=4, sort_keys=False)
    if args.alg == 'A':
        dataset_train = MyDataset_A(args.f_train, e2e_dictionary, tokenizer, args.seed)
        dataset_valid = MyDataset_A(args.f_valid, e2e_dictionary, tokenizer, args.seed)
        dataset_test  = MyDataset_A(args.f_test, e2e_dictionary, tokenizer, args.seed)
    else:
        dataset_train = MyDataset_O(args.f_train, e2e_dictionary, tokenizer, args.seed)
        dataset_valid = MyDataset_O(args.f_valid, e2e_dictionary, tokenizer, args.seed)
        dataset_test  = MyDataset_O(args.f_test, e2e_dictionary, tokenizer, args.seed)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch, shuffle=False)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch, shuffle=False)

    # (3) model settings
    print('(3) network settings')
    nlg_encoder = NLG_Encoder(e2e_dictionary['mr']['dim'], int(args.hid_dim), int(args.enc_layer), int(args.enc_head), int(args.enc_pf_dim), args.dropout, e2e_dictionary['mr']['max_num_token'], device)
    nlg_decoder = NLG_Decoder(e2e_dictionary['txt']['dim'], int(args.hid_dim), int(args.dec_layer), int(args.dec_head), int(args.dec_pf_dim), args.dropout, e2e_dictionary['txt']['max_num_token'], device)
    nlg_model = NLG_Model(nlg_encoder, nlg_decoder, e2e_dictionary['mr']['s2i']['<pad>'], e2e_dictionary['txt']['s2i']['<pad>'], device)
    nlg_model = nlg_model.to(device)
    nlg_model.apply(initialize_weights);
    print('The model (NLG) has {} trainable parameters'.format(count_parameters(nlg_model)))

    # (4) training settings
    print('(4) training settings')
    optimizer = torch.optim.Adam(nlg_model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index = e2e_dictionary['txt']['s2i']['<pad>'])

    # (5) save parameters
    print('(5) save parameters')
    parameters = {
        'corpus': {
            'train': args.f_train,
            'valid': args.f_valid,
            'test': args.f_test
        },
        'd_model': args.d_model,
        'parameters': count_parameters(nlg_model),
        'training': {
            'epoch': args.epoch,
            'batch': args.batch,
            'lr': args.lr,
            'dropout': args.dropout,
            'clip': args.clip,
            'seed': args.seed
        },
        'transformer': {
            'dim_hid': args.hid_dim,
            'encoder': {
                'n_layer': args.enc_layer,
                'n_head': args.enc_head,
                'dim_pf': args.enc_pf_dim
            },
            'decoder': {
                'n_layer': args.dec_layer,
                'n_head': args.dec_head,
                'dim_pf': args.dec_pf_dim
            }
        },
        'algorithm': args.alg
    }
    with open(args.d_model.rstrip('/')+'/parameter.json', 'w', encoding='utf-8') as f:
        json.dump(parameters, f, ensure_ascii=False, indent=4, sort_keys=True)

    # (6) trainnig
    print('(6) training')
    best_epoch = 0
    best_loss_valid = float('inf')
    a_performance = {
        'loss_train': [],
        'loss_valid': [],
        'loss_test': [],
        'datetime': [],
        'current_epoch': 0,
        'best_epoch': best_epoch,
        'best_loss_valid': best_loss_valid
    }
    for epoch in range(args.epoch):
        print('epoch: {} begin ...'.format(epoch))
        start_time = time.time()
        loss_train = train.train_nlg(nlg_model, dataloader_train, optimizer, criterion, args.clip, device, args.v)
        loss_valid = train.evaluate_nlg(nlg_model, dataloader_valid, criterion, device)
        loss_test = train.evaluate_nlg(nlg_model, dataloader_test, criterion, device)

        if best_loss_valid > loss_valid:
            best_loss_valid = loss_valid
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_loss_valid': best_loss_valid,
                'optimizer_dict': optimizer.state_dict(),
                'model_dict': nlg_model.state_dict(),
                'model': nlg_model},
                       args.d_model.rstrip('/')+'/best.dat')
        torch.save({
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_loss_valid': best_loss_valid,
            'optimizer_dict': optimizer.state_dict(),
            'model_dict': nlg_model.state_dict(),
            'model': nlg_model},
                   args.d_model.rstrip('/')+'/model_'+str(epoch).zfill(3)+'.dat')
        a_performance['loss_train'].append(loss_train)
        a_performance['loss_valid'].append(loss_valid)
        a_performance['loss_test'].append(loss_test)
        a_performance['datetime'].append(datetime.datetime.now().isoformat())
        a_performance['current_epoch'] = epoch
        a_performance['best_epoch'] = best_epoch
        a_performance['best_loss_valid'] = best_loss_valid
        with open(args.d_model+'/performance.json', 'w', encoding='utf-8') as f:
            json.dump(a_performance, f, ensure_ascii=False, indent=4, sort_keys=True)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print('epoch: {} end | Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
        print('\ttrain loss: {} | train PPL: {}'.format(loss_train, math.exp(loss_train)))
        print('\tvalid loss: {} | valid PPL: {}'.format(loss_valid, math.exp(loss_valid)))
        print('\ttest  loss: {} | test  PPL: {}'.format(loss_test, math.exp(loss_test)))

    print('** done **')
