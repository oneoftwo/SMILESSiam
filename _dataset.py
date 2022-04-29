from torch.utils.data import Dataset
from util.process_mol.mol_to_seq import get_random_smiles, get_all_possible_smiles
from rdkit import Chem
import numpy as np
import torch
import random
from copy import deepcopy


class SMILESDataset(Dataset):
    def __init__(self, sample_list, c_to_i, is_target=False):
        super().__init__()
        
        self.is_target = is_target
        self.sample_list = sample_list

        assert not 'q' in c_to_i, 'c_to_i assertion error'
        assert not 'Q' in c_to_i, 'c_to_i assertion error' 
        self.c_to_i = c_to_i + ['Q'] + ['q'] # add SOS/EOS token
        self.n_char = len(self.c_to_i)

    def __len__(self):
        l = len(self.sample_list)
        return l 
    
    def __getitem__(self, idx):
        sample = {}
        sample_1 = self._pre_getitem(idx) # augment 1 
        sample_2 = self._pre_getitem(idx) # augment 2
        sample['seq_1'] = sample_1['seq']
        sample['seq_2'] = sample_2['seq']
        sample['length_1'] = sample_1['length']
        sample['length_2'] = sample_2['length']
        sample['smiles_1'] = sample_1['smiles']
        sample['smiles_2'] = sample_2['smiles']
        if self.is_target:
            sample['target'] = sample_1['target']

        try:
            sample['pp'] = self.sample_list[idx]['pp']
            # sample['fp'] = self.sample_list[idx]['fp'] #1024
        except:
            pass

        return sample
        
    def _pre_getitem(self, idx):
        sample = deepcopy(self.sample_list[idx])
        mol = Chem.MolFromSmiles(sample['smiles'])
        
        """
        for _ in range(random.randint(1, 100)): # random seed problem
            Chem.MolToSmiles(Chem.MolFromSmiles('O'), doRandom=True)
        try:
            smiles = Chem.MolToSmiles(mol, doRandom=True)
        except:
            print(sample['smiles'])
            print('problem occured while getitem')
        """
        try:
            smiles = get_random_smiles(sample['smiles'])
            # smiles = sample['smiles']
        except:
            smiles = sample['smiles']
            print(f'getitem failed to get random smiles: {smiles}')
            smiles = sample['smiles']

        seq = self._smiles_to_seq(smiles)
        sample['seq'] = seq
        sample['length'] = len(seq)
        sample['smiles'] = smiles
        return sample
    
    def get_n_char(self):
        return self.n_char

    def _char_to_idx(self, char):
        idx = self.c_to_i.index(char)
        return idx

    def _smiles_to_seq(self, smiles):
        # additionally, add SOS / EOS token (Q, q)
        smiles = 'Q' + smiles + 'q'
        seq = []
        for char in smiles:
            idx = self._char_to_idx(char)
            seq.append(int(idx))
        seq = torch.Tensor(seq)
        return seq
            

def seq_collate_fn(batch):
    sample  = {}
    # seq
    # length 
    sample['length_1'] = torch.Tensor([x['length_1'] for x in batch])
    sample['length_2'] = torch.Tensor([x['length_2'] for x in batch])
    # smiles 
    sample['smiles_1'] = [x['smiles_1'] for x in batch]
    sample['smiles_2'] = [x['smiles_2'] for x in batch]

    sample['seq_1'] = \
            torch.nn.utils.rnn.pad_sequence([x['seq_1'] for x in batch], \
            batch_first=True, padding_value=0)
    sample['seq_2'] = \
            torch.nn.utils.rnn.pad_sequence([x['seq_2'] for x in batch], \
            batch_first=True, padding_value=0)
    try:
        sample['target'] = torch.Tensor([x['target'] for x in batch])
    except:
        pass

    try:
        sample['pp'] = torch.Tensor([list(x['pp']) for x in batch])
    except:
        pass

    return sample


if __name__ == '__main__':
    smiles = 'CCOC.CCC'
    mol = get_random_smiles(smiles)
    print(mol)
        
