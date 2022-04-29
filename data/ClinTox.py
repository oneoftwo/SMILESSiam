import pickle
from dgllife.data import ClinTox
from _util_data import process_smiles_target_pair_list


dataset = ClinTox()
n_tasks = len(dataset[0][2])

print()
print(f'number of tasks: {n_tasks}')

for task_idx in range(n_tasks):
    
    print()
    print(f'# processing task number {task_idx}')

    smiles_target_pair_list = []
    for x in dataset:
        smiles = x[0]
        target = int(x[2][task_idx].squeeze().item())
        mask = int(x[3][task_idx].squeeze().item())
        if mask == 1:
            if not '*' in smiles:
                smiles_target_pair_list.append([smiles, target])
    
    print('number of raw data: ', len(smiles_target_pair_list))

    sample_list = process_smiles_target_pair_list(smiles_target_pair_list, n_process=16)

    print('number of processed data: ', len(sample_list))

    save_fn = f'./ClinTox/ClinTox_{task_idx}.pkl'

    pickle.dump(sample_list, open(save_fn, 'wb'))

    print(f'saved at {save_fn}')

