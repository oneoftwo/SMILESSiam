import torch 
from dgllife.data import HIV
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
import sys
import os
from rdkit import Chem 
from torch_geometric.data import Data
import pickle

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from util.process_mol import mol_to_seq 


dataset = HIV(smiles_to_bigraph, CanonicalAtomFeaturizer())

sample_list = []
for x in dataset:
    sample = {}
    smiles = x[0]
    labels = x[2]
    masks = x[3]
    sample['smiles'] = smiles
    if masks[0] == 1:
        sample['target'] = int(labels[0].item())
    sample_list.append(sample)
print('number of raw data: ', len(sample_list))

data = []
for sample in sample_list:
    try:
        smiles = sample['smiles']
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

        assert not '.' in smiles
        
        data.append(sample)
        print(sample)

    except:
        print(sample)

print('number of processed data: ', len(data))
pickle.dump(data, open('./HIV/HIV.pkl', 'wb'))
print('saved at HIV/HIV.pkl')


