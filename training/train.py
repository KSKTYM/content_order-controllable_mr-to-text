#! python
# -*- coding: utf-8 -*-

import torch

##
## NLU
##
def train_nlu(model, iterator, optimizer, criterion, clip, device, verbose_flag):
    model.train()
    train_loss = 0
    for i, (mr_value, txt) in enumerate(iterator):
        mr_value = mr_value.to(device)
        txt = txt.to(device)
        if verbose_flag is True:
            print('***** train i : '+str(i)+' *****')
            # mr_value [batch_size(128), max_num_token_mr(10)]
            print('(1) mr_value: '+str(mr_value.size()))
            print(mr_value)
            # txt [batch_size(128), max_num_token_txt(88)]
            print('(1) txt: '+str(txt.size()))
            print(txt)

        optimizer.zero_grad()
        output_mr_value, _ = model(txt, mr_value[:,:-1])
        if verbose_flag is True:
            # output_mr_value [batch_size(128), max_num_token_mr-1(9), mr_dim(36)]
            print('(2) output_mr_value: '+str(output_mr_value.size()))
            print(output_mr_value)

        output_mr_value_dim = output_mr_value.shape[-1]
        output_mr_value = output_mr_value.contiguous().view(-1, output_mr_value_dim)
        if verbose_flag is True:
            # output_mr_value [batch_size*(max_num_token_mr-1), output_mr_value_dim] (1152, 36)
            print('(3) output_mr_value: '+str(output_mr_value.size()))
            print(output_mr_value)

        mr_value = mr_value[:,1:].contiguous().view(-1)
        if verbose_flag is True:
            # mr_value [batch_size*(max_num_token_mr-1)] (1152)
            print('(4) mr_value:'+str(mr_value.size()))
            print(mr_value)

        loss = criterion(output_mr_value, mr_value)
        if verbose_flag is True:
            # loss [1]
            print('(5) loss: '+str(loss.size()))
            print(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        
    return train_loss / len(iterator)

def evaluate_nlu(model, iterator, criterion, device):
    model.eval()
    evaluate_loss = 0
    with torch.no_grad():
        for i, (mr_value, txt) in enumerate(iterator):
            mr_value = mr_value.to(device)
            txt = txt.to(device)
            output_mr_value, _ = model(txt, mr_value[:,:-1])
            output_mr_value_dim = output_mr_value.shape[-1]
            output_mr_value = output_mr_value.contiguous().view(-1, output_mr_value_dim)
            mr_value = mr_value[:,1:].contiguous().view(-1)
            loss = criterion(output_mr_value, mr_value)
            evaluate_loss += loss.item()
    return evaluate_loss / len(iterator)


##
## NLG
##
def train_nlg(model, iterator, optimizer, criterion, clip, device, verbose_flag):
    model.train()
    train_loss = 0
    for i, (mr_value, txt) in enumerate(iterator):
        mr_value = mr_value.to(device)
        txt = txt.to(device)
        if verbose_flag is True:
            print('***** train i : '+str(i)+' *****')
            # mr_value: [batch_size(128), max_num_token_mr(10/8)]
            print('(1) mr_value: '+str(mr_value.size()))
            print(mr_value)
            # txt: [batch_size(128), max_num_token_txt(88)]
            print('(1) txt: '+str(txt.size()))
            print(txt)

        optimizer.zero_grad()
        output_txt, _ = model(mr_value, txt[:,:-1])
        if verbose_flag is True:
            # output_txt: [batch_size(128), max_num_token_txt-1(87), txt_dim(2875)]
            print('(2) output_txt: '+str(output_txt.size()))
            print(output_txt)

        output_txt_dim = output_txt.shape[-1]
        output_txt = output_txt.contiguous().view(-1, output_txt_dim)
        if verbose_flag is True:
            # output_txt: [batch_size*(max_num_token_txt-1)(11136), txt_dim(2875)]
            print('(3) output_txt: '+str(output_txt.size()))
            print(output_txt)

        txt = txt[:,1:].contiguous().view(-1)
        if verbose_flag is True:
            # txt: [batch_size*(max_num_token_txt-1)(11136)]
            print('(4) txt:'+str(txt.size()))
            print(txt)
            
        loss = criterion(output_txt, txt)
        if verbose_flag is True:
            print('(5) loss: '+str(loss.size()))
            print(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        
    return train_loss / len(iterator)

def evaluate_nlg(model, iterator, criterion, device):
    model.eval()
    evaluate_loss = 0
    with torch.no_grad():
        for i, (mr_value, txt) in enumerate(iterator):
            mr_value = mr_value.to(device)
            txt = txt.to(device)
            output_txt, _ = model(mr_value, txt[:,:-1])
            output_txt_dim = output_txt.shape[-1]
            output_txt = output_txt.contiguous().view(-1, output_txt_dim)
            txt = txt[:,1:].contiguous().view(-1)
            loss = criterion(output_txt, txt)
            evaluate_loss += loss.item()
    return evaluate_loss / len(iterator)
