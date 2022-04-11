import numpy as np 
import pandas as pd 
import pickle
from rdkit import Chem
from tqdm import tqdm
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


""" 
AR, AhR, AR-LBD, ER, ER-LBD, aromatase, PPAR-gamma, ARE, ATAD5, HSE, MMP, p53
"""


global tox_key_list
tox_key_list = ['NR-AR', 'NR-AhR', 'NR-AR-LBD', 'NR-ER-LBD', 'NR-Aromatase', \
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']


def get_mol_feature(mol):
    global tox_key_list
    sample = {}
    try:
        smiles = Chem.MolToSmiles(mol)
        sample = {}
        for tox_key in tox_key_list:
            try:
                label = mol.GetProp(tox_key)
                sample[tox_key] = int(label)
            except:
                sample[tox_key] = -1
        sample['smiles'] = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        sample = None
    return sample
        

def preprocess_tox21(fn):
    global tox_key_list
    suppl = Chem.SDMolSupplier(fn)
    
    data_list = []
    for mol in suppl:
        sample = get_mol_feature(mol)
        if not sample == None:
            data_list.append(sample)
    print(len(data_list))

    to_save_dict = {}
    for tox_key in tox_key_list:
        to_save = []
        for data in tqdm(data_list):
            sample = {}
            if not data[tox_key] == -1:
                try:
                    y = data[tox_key]
                    y = torch.Tensor([y]).unsqueeze(0)
                    sample['smiles'] = data['smiles']
                    
                    assert not '.' in sample['smiles']

                    mol = Chem.MolFromSmiles(data['smiles'])
                    mol = Chem.RemoveHs(mol)
                    sample['target'] = data[tox_key]
                    to_save.append(sample)

                except:
                    pass
        print(f'{tox_key}: {len(to_save)}')
        pickle.dump(to_save, open(f'./tox21/tox21_{tox_key}.pkl', 'wb'))
    return data_list


if __name__ == '__main__':
    fn = './tox21/tox21.sdf'
    preprocess_tox21(fn)

