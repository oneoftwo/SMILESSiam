import numpy as np
import random
import pickle
import os
from copy import deepcopy
from rdkit import Chem
import torch 
from torch import optim 
from torch import nn
from _dataset import SMILESDataset, seq_collate_fn
from tqdm import tqdm
import _train as TRAIN
import _argument as ARGUMENT
from util.sys.sys import set_cuda_visible_devices
from util.train.evaluate import count_learnable_parameters
from _model import SMILESSiam, LanguageModel
from torch.utils.data import DataLoader


def main():
    
    print('\ntrain_siam.py\n')
    print('#' * 80 + '\n')
    
    set_cuda_visible_devices(is_print=True, ngpus=1)
    print()
    # _ = torch.rand(1).cuda()
    args = ARGUMENT.get_train_siam_args()
    
    for x in vars(args):
        print(f'{x}: {vars(args)[x]}')
    print()

    # load
    data_fn = args.data_fn
    sample_list = pickle.load(open(data_fn, 'rb'))
    random.shuffle(sample_list)
     
    print(f'total loaded data: {len(sample_list)}')
    
    ori_c_to_i = pickle.load(open('./data/c_to_i.pkl', 'rb'))
    train_set = SMILESDataset(sample_list[:int(len(sample_list) * 0.9)], ori_c_to_i)
    valid_set = SMILESDataset(sample_list[int(len(sample_list) * 0.9):], ori_c_to_i)
    c_to_i = train_set.c_to_i

    print(f'train data: {len(train_set)}\nval data: {len(valid_set)}\n')
    # print(f'data sample: {train_set[0]}\n')
    print(c_to_i, '\n')

    # define model
    representation_model = LanguageModel(n_char=len(c_to_i), hid_dim=args.hid_dim, n_layer=args.n_layer)
    siam_model = SMILESSiam(representation_model, use_pp_prediction=args.use_pp_prediction)

    print(f'{siam_model}\n')

    best_val_loss = 1e10
            
    print(f"|{'epoch':^8}|" + 
           f"{'train_siam':^12}|" +
           f"{'train_pp':^12}|" +
           f"{'val_siam':^12}|" + 
           f"{'val_pp':^12}|" +
           f"{'val_z_std':^12}|" +
           f"{'time':^12}|" + 
           f"{'lr * 1e-4':^12}|")
    
    train_loader = DataLoader(train_set, batch_size = args.bs, shuffle=True, \
            collate_fn=seq_collate_fn, num_workers=8)
    val_loader = DataLoader(valid_set, batch_size=args.bs, shuffle=True, \
            collate_fn=seq_collate_fn, num_workers=8)
    
    # train model and validate
    lr = args.lr
    for epoch in range(1, args.bs + 1):
        optimizer = optim.SGD(siam_model.parameters(), lr=lr)
        siam_model, train_result = \
                TRAIN.process_siam(siam_model, train_loader, optimizer=optimizer, args=args)
        _, val_result = \
                TRAIN.process_siam(siam_model, val_loader, args=args)
        lr = lr * args.lr_decay
        
        # save best model
        marker = ''
        if args.save_dir:
            siam_model.cpu()
            torch.save(siam_model.state_dict(), f'{args.save_dir}model_{epoch}.pt')
        if val_result['loss_pp'] + val_result['loss_siam']  < best_val_loss:
            if args.save_dir:
                    siam_model.cpu()
                    torch.save(siam_model.state_dict(), f'{args.save_dir}model_best.pt')
            marker = '*'
            best_val_loss = val_result['loss_pp'] + val_result['loss_siam']
        else:
            marker = ''
        
        # print
        print(f"|{epoch:^8}|" + 
                f"{train_result['loss_siam']:^12.4f}|" +
                f"{train_result['loss_pp']:^12.4f}|" +
                f"{val_result['loss_siam']:^12.4f}|" + 
                f"{val_result['loss_pp']:^12.4f}|" + 
                f"{val_result['std']:^12.4f}|" +
                f"{train_result['time'] + val_result['time']:^12.2f}|" + 
                f"{lr * 1e4:^12.4f}|"
                f"{marker}")


if __name__ == '__main__':
    main()

