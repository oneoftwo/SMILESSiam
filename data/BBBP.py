import pickle
import numpy as np
from dgllife.data import BBBP
from _util_data import process_smiles_target_pair_list


dataset = BBBP()

smiles_target_pair_list = []
for x in dataset:
    smiles = x[0]
    labels = int(x[2].squeeze().item())
    masks = x[3]
    if masks[0] == 1:
        smiles_target_pair_list.append([smiles, labels])
    
print('number of raw data: ', len(smiles_target_pair_list))

sample_list = process_smiles_target_pair_list(smiles_target_pair_list, n_process=16)

print('number of processed data: ', len(sample_list))

pickle.dump(sample_list, open('./BBBP/BBBP.pkl', 'wb'))

