import random
import math 
import numpy as np 
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem.AllChem import GetAdjacencyMatrix
from rdkit.Chem.Descriptors import MaxPartialCharge, MinPartialCharge, \
        MolWt, NumValenceElectrons, NumRadicalElectrons, qed, TPSA 
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.Lipinski import NumRotatableBonds, NumHAcceptors, \
        NumHDonors, NumHeteroatoms, NumAliphaticCarbocycles, \
        NumAliphaticHeterocycles, NumAliphaticRings, \
        NumAromaticCarbocycles, NumAromaticHeterocycles, \
        NumAromaticRings, NumSaturatedCarbocycles, \
        NumSaturatedHeterocycles, NumSaturatedRings 


def mol_to_physical_property(mol):
    pq = []
    # Charge
    #pq.append(MaxPartialCharge(mol))
    #pq.append(MinPartialCharge(mol))
    # Fragment
    pq.append(NumAliphaticCarbocycles(mol))
    pq.append(NumAliphaticHeterocycles(mol))
    pq.append(NumAliphaticRings(mol))
    pq.append(NumAromaticCarbocycles(mol))
    pq.append(NumAromaticHeterocycles(mol))
    pq.append(NumAromaticRings(mol))
    pq.append(NumSaturatedCarbocycles(mol))
    pq.append(NumSaturatedHeterocycles(mol))
    pq.append(NumSaturatedRings(mol))
    pq.append(NumHDonors(mol))
    pq.append(NumHAcceptors(mol))
    pq.append(NumRotatableBonds(mol))
    pq.append(NumHeteroatoms(mol))
    # DrugLikeness
    pq.append(MolLogP(mol))
    pq.append(MolWt(mol))
    pq.append(MolMR(mol))
    pq.append(TPSA(mol))
    pq.append(qed(mol))
    # Reactivity
    pq.append(NumValenceElectrons(mol))
    pq.append(NumRadicalElectrons(mol))
    pq = scaling(pq, \
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                    -5.9718, 52.036, 5.163, 0.0, 0.08327133915786711, 16.0, \
                    0.0], [9.0, 13.0, 13.0, 2.0, 3.0, 3.0, 9.0, 13.0, 13.0, \
                    5.0, 9.0, 9.0, 13.0, 7.4363, 147.952, 89.257, 158.88, \
                    0.7146923410153729, 68.0, 5.0])
    return pq


def scaling(arr, min, diff):
    arr = np.array(arr)
    min = np.array(min)
    diff = np.array(diff)
    scaled_arr = (arr - min) / diff
    return np.clip(scaled_arr, 0, 1)


if __name__ == '__main__':
    pass

