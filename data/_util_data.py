import sys
import os
import numpy as np
from rdkit import Chem 
from rdkit.Chem import AllChem
from multiprocessing import Pool 
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util.process_mol.mol_property import mol_to_physical_property 
from util.process_mol.mol_to_seq import get_random_smiles


def process_smiles_target_pair(smiles_target_pair):
    smiles, target = smiles_target_pair
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        mol = Chem.MolFromSmiles(smiles)
        get_random_smiles(smiles, randint=100)
        pp = mol_to_physical_property(Chem.MolFromSmiles(smiles))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_short = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
        
        # exceptions!
        assert not 'X' in smiles # Xe in PubChem
        assert not 'Q' in smiles # SOS
        assert not 'q' in smiles # EOS 

        sample = {}
        sample['smiles'] = smiles 
        sample['pp'] = pp # np array [1 20]
        sample['fp'] = np.array(fp) # [1024] 
        sample['fp_short'] = np.array(fp_short) # [256]
        sample['target'] = target
    except:
        sample = None
    return sample
    

def process_smiles_target_pair_list(smiles_target_pair_list, n_process=16): # multiprocessing added 
    pool = Pool(n_process)
    sample_list = pool.map(process_smiles_target_pair, smiles_target_pair_list)
    sample_list = [x for x in sample_list if x != None]
    return sample_list
    

if __name__ == '__main__':
    # a = process_smiles_target_pair_list([('CC', 1),('COCC', 1)], n_process=2)
    # print(a)
    import torch 
    a = torch.Tensor([1])
    print(a)
    print(int(a.squeeze()))

