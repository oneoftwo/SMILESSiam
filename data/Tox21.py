import numpy as np 
from rdkit import Chem
import pickle
from _util_data import process_smiles_target_pair_list


tox_key_list = ['NR-AR', 'NR-AhR', 'NR-AR-LBD', 'NR-ER-LBD', 'NR-Aromatase', \
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'NR-ER']

suppl = Chem.SDMolSupplier('./Tox21/Tox21.sdf')
i = 0
for task_name in tox_key_list:
    
    print()
    print(task_name)
    
    smiles_target_pair_list = []
    for mol in suppl:
        try:
            smiles = Chem.MolToSmiles(mol)
            target = int(mol.GetProp(task_name))
            smiles_target_pair_list.append([smiles, target])
        except:
            pass
        
    print(f'num raw data: {len(smiles_target_pair_list)}')
    
    sample_list = process_smiles_target_pair_list(smiles_target_pair_list, n_process=20)

    print(f'number of processed data: , {len(sample_list)}')

    save_fn = f'./Tox21/Tox21_{i}.pkl'
    i += 1
    
    pickle.dump(sample_list, open(save_fn, 'wb'))
    

