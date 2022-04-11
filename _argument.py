import argparse
import os
import shutil
from util.sys import io


def get_train_siam_args():
    parser = argparse.ArgumentParser()
    
    # dir and fn
    parser.add_argument('--data_fn', type=str, required=False, 
            default='./data/PubChem/PubChem_10000.pkl')
    parser.add_argument('--save_dir', type=str, required=False,
            default=False)
    
    # model hyperparameters 
    parser.add_argument('--hid_dim', type=int, required=False,
            default=128)
    parser.add_argument('--n_layer', type=int, required=False,
            default=3)

    # train hyperparametes
    parser.add_argument('--bs', type=int, required=False,
            default=256)
    parser.add_argument('--lr', type=float, required=False,
            default=1e-5)
    parser.add_argument('--n_epoch', type=int, required=False,
            default=1000)
    parser.add_argument('--lr_decay', type=float, required=False,
            default=1.0)

    parser.add_argument('--use_pp_prediction', required=False,
            action='store_true', default=False)
    parser.add_argument('--pp_loss_ratio', type=float, required=False,
            default=1.0)

    args = parser.parse_args()
    
    if not args.save_dir == False:
        if args.save_dir[-1] != '/':
            args.save_dir += '/'
    
    return args


def get_train_clf_args():
    parser = argparse.ArgumentParser()
    
    # dir and fn
    parser.add_argument('--data_fn', type=str, required=False, 
            default='./data/HIV/HIV.pkl')
    parser.add_argument('--save_dir', type=str, required=False,
            default=False)
    
    # model hyperparameters 
    parser.add_argument('--hid_dim', type=int, required=False,
            default=128)
    parser.add_argument('--n_layer', type=int, required=False,
            default=3)

    # train hyperparametes
    parser.add_argument('--bs', type=int, required=False,
            default=128)
    parser.add_argument('--lr', type=float, required=False,
            default=1e-4)
    parser.add_argument('--n_epoch', type=int, required=False,
            default=1000)
    parser.add_argument('--lr_decay', type=float, required=False,
            default=0.99)

    args = parser.parse_args()
    
    if not args.save_dir == False:
        if args.save_dir[-1] != '/':
            args.save_dir += '/'
    
    return args


