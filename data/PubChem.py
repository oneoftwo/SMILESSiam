import numpy as np 
import pandas as pd 
import pickle
from rdkit import Chem
from tqdm import tqdm
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util.process_mol.mol_property import mol_to_physical_property

def preprocess_pubchem(fn, save_fn, num_to_get=10000):
    """ 
    graph sample list:
        target: physical mol_property
        
    """
    f = open(fn, 'r')
    data = []
    line_list = []
    for line in f:
        if len(line_list) < num_to_get * 2:
            line_list.append(line)
        else:
            break

    for idx, line in enumerate(tqdm(line_list)):
        try:
            assert len(data) < num_to_get

            line = line.strip().split()
            sample = {}
            
            smiles = line[1]
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            
            assert not '.' in smiles
            
            sample['smiles'] = smiles
            sample['id'] = line[0]
            
            mol = Chem.MolFromSmiles(smiles)
            sample['pp'] = mol_to_physical_property(mol)

            data.append(sample)
        except:
            print(line)

    print(len(data))
    
    pickle.dump(data, open(save_fn, 'wb'))
    
    print(f'saved at {save_fn}')
    
    return None


if __name__ == '__main__':
    import sys 
    fn = './PubChem/PubChem.txt'
    num_to_get = int(sys.argv[1])
    save_fn = f'./PubChem/PubChem_{num_to_get}.pkl'
    preprocess_pubchem(fn, save_fn, num_to_get=num_to_get)
    
