import argparse 
from train_clf_control import main
# from train_clf import main 


exp_name_list = ['BACE', 'ClinTox', 'HIV', 'MUV', 'SIDER', 'Tox21', 'BBBP']

exp_list = []
# exp_list.append(['./data/BACE/BACE.pkl'])
# exp_list.append([f'./data/ClinTox/ClinTox_{n}.pkl' for n in range(2)])
# exp_list.append(['./data/HIV/HIV.pkl'])
exp_list.append([f'./data/MUV/MUV_{n}.pkl' for n in range(17)])
exp_list.append([f'./data/SIDER/SIDER_{n}.pkl' for n in range(27)])
# exp_list.append([f'./data/Tox21/Tox21_{n}.pkl' for n in range(12)])
exp_list.append(['./data/BBBP/BBBP.pkl'])

print(exp_name_list)

parser = argparse.ArgumentParser()
args = parser.parse_args()

# fixed parameters
# model hyper param 
args.hid_dim = 256
args.n_layer = 3
# train hyper 
args.bs = 32
args.lr = 1e-5
args.n_epoch = 1000
args.lr_decay = 0.99

# experiment 
args.use_pp_prediction = True

for exp in exp_list:
    print(f'! {exp}')
    
    n_data_fn = len(exp)
    avg_loss, avg_acc, avg_auc_roc, avg_auc_prc = 0, 0, 0, 0

    result_list = []
    for data_fn in exp:
        args.data_fn = data_fn 
        best_result = main(args)
        result_list.append(best_result)
        print()
        print(data_fn)
        print(best_result)
        print()

    for result in result_list:
        avg_loss += result['loss'] / n_data_fn 
        avg_acc += result['accuracy'] / n_data_fn 
        avg_auc_roc += result['auc_roc'] / n_data_fn 
        avg_auc_prc += result['auc_prc'] / n_data_fn
    
    print()
    print(exp)
    print('!!!result!!!\n, loss, acc, auc_roc, auc_prc\n')
    print(avg_loss)
    print(avg_acc)
    print(avg_auc_roc)
    print(avg_auc_prc)
    print()
    

