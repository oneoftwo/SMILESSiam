#!/bin/bash
#PBS -N LJW_rep
#PBS -l nodes=gnode1:ppn=4
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

exp_name=siam

source activate LJW_DeepSLIP
rm -rf ./save/${exp_name}/ 
mkdir ./save/${exp_name}/ 

python -u train_siam.py \
--data_fn ./data/PubChem/PubChem_preprocessed_1000000.pkl \
--save_dir ./save/${exp_name}/ \
--bs 256 \
--lr 1e-8 \
--n_epoch 1000 \
1>./save/${exp_name}/output.txt \
2>./save/${exp_name}/error.txt

