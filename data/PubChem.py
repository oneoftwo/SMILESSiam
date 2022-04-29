import pickle

from _util_data import process_smiles_target_pair_list


def preprocess_pubchem(fn, save_fn, num_to_get=10000):
    f = open(fn, 'r')
    
    # get smiles (2* of num_to_get)
    smiles_target_pair_list = []
    for line in f:
        if len(smiles_target_pair_list) < num_to_get * 2:
            smiles_target_pair_list.append([line.strip().split()[1], None])
        else:
            break

    sample_list = process_smiles_target_pair_list(smiles_target_pair_list, n_process=20)[:num_to_get]

    pickle.dump(sample_list, open(save_fn, 'wb'))
    
    print(f'saved at {save_fn}')
    
    return None


if __name__ == '__main__':
    import sys 
    fn = './PubChem/PubChem.txt'
    num_to_get = int(sys.argv[1])
    save_fn = f'./PubChem/PubChem_{num_to_get}.pkl'
    preprocess_pubchem(fn, save_fn, num_to_get=num_to_get)
    
