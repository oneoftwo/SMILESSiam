#!/bin/bash
#PBS -N LJW_SS_pp
#PBS -l nodes=gnode7:ppn=4
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

exp_name=siam_pp_L

source activate LJW_DeepSLIP
rm -rf ./save/${exp_name}/ 
mkdir ./save/${exp_name}/ 

python -u train_siam.py \
--data_fn ./data/PubChem/PubChem_1000000.pkl \
--save_dir ./save/${exp_name}/ \
--use_pp_prediction \
--pp_loss_ratio 1.0 \
--hid_dim 256 \
--n_layer 3 \
--bs 512 \
--lr 1e-6 \
--n_epoch 1000 \
1>./save/${exp_name}/output.txt \
2>./save/${exp_name}/error.txt

