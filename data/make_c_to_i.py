import numpy as np 
import pandas as pd 
import pickle
from rdkit import Chem
from tqdm import tqdm
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from multiprocessing import Pool
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from util.process_mol.mol_to_seq import get_random_smiles


def run(smiles_list):
    c_to_i = []
    for smiles in tqdm(smiles_list):
        
        original_smiles = smiles
        
        mol = Chem.MolFromSmiles(smiles)
        for _ in range(10):
            smiles = get_random_smiles(smiles)
            for char in smiles:
                if not char in c_to_i:
                    c_to_i.append(char)
    return c_to_i


def main(n_process=16):
    
    print('\nmake_c_to_i.py\n')

    fn_list = sys.argv[1:] # pkl files
    
    c_to_i = []
    
    total_smiles_list = []
    for fn in fn_list:
        sample_list = pickle.load(open(fn, 'rb'))
        total_smiles_list += [x['smiles'] for x in sample_list]
    
    print(len(total_smiles_list))
    
    pool_input_list, temp = [[] for _ in range(n_process)], []
    for idx, smiles in enumerate(total_smiles_list):
        pool_input_list[idx % n_process].append(smiles)

    pool = Pool(n_process)
    c_to_i_list = pool.map(run, pool_input_list)
    
    # merge c to is
    total_c_to_i = []
    for c_to_i in c_to_i_list:
        for char in c_to_i:
            if not char in total_c_to_i:
                total_c_to_i.append(char)
    c_to_i = total_c_to_i
    
    pickle.dump(c_to_i, open('./c_to_i.pkl', 'wb'))
    
    print(c_to_i)
    print(len(c_to_i))
    print()

    return None 


if __name__ == '__main__':
    main(n_process=20)
    
