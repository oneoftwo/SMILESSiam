#!/bin/bash
#PBS -N LJW_SS_siam_HIV_all
#PBS -l nodes=gnode7:ppn=4
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`


exp_name=exp_0_control_HIV

source activate LJW_DeepSLIP
rm -rf ./save/${exp_name}/ 
mkdir ./save/${exp_name}/ 

python -u ./train_clf_control.py --lr 1e-5 --bs 128 --data_fn ./data/HIV/HIV.pkl --hid_dim 256 --n_layer 3\
1>./save/${exp_name}/output.txt \
2>./save/${exp_name}/error.txt


exp_name=exp_0_siam_HIV

source activate LJW_DeepSLIP
rm -rf ./save/${exp_name}/ 
mkdir ./save/${exp_name}/ 

python -u ./train_clf.py --lr 1e-5 --bs 128 --data_fn ./data/HIV/HIV.pkl --hid_dim 256 --n_layer 3 --siam_model_fn ./save/siam_L/model_best.pt \
1>./save/${exp_name}/output.txt \
2>./save/${exp_name}/error.txt


exp_name=exp_0_siam_pp_HIV

source activate LJW_DeepSLIP
rm -rf ./save/${exp_name}/ 
mkdir ./save/${exp_name}/ 

python -u ./train_clf.py --lr 1e-5 --bs 128 --data_fn ./data/HIV/HIV.pkl --hid_dim 256 --n_layer 3 --siam_model_fn ./save/siam_pp_L/model_best.pt --use_pp_prediction \
1>./save/${exp_name}/output.txt \
2>./save/${exp_name}/error.txt

