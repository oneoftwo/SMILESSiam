#!/bin/bash
#PBS -N LJW_SS_bcm
#PBS -l nodes=gnode1:ppn=16
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`


exp_name=exp_0_pp_bcm


source activate LJW_DeepSLIP
source activate LJW_add

rm -rf ./save/${exp_name}/ 
mkdir ./save/${exp_name}/ 

python -u benchmark_clf.py \
1>./save/${exp_name}/output.txt \
2>./save/${exp_name}/error.txt

