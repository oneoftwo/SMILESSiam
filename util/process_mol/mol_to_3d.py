import numpy as np
import time
import random
import tqdm
from copy import deepcopy

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


def optimize_mol_position(mol, n_conformer=20, iteration=2000):
    """   
    returns None if not converged 
    returns optimized mol object when converged
    """
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, n_conformer)
    confs = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=iteration)
    idx, min_e = -1, 1e30
    for i, conf in enumerate(confs):
        converged_indicator, energy = conf
        if converged_indicator == 0 and min_e > energy:
            idx, min_e = i, energy
    if idx == -1:
        return None
    optimized_conf = mol.GetConformer(idx)
    mol_out = deepcopy(mol)
    AllChem.EmbedMolecule(mol_out)
    for atom_idx in range(mol.GetNumAtoms()):
        position = optimized_conf.GetAtomPosition(atom_idx)
        mol_out.GetConformer().SetAtomPosition(atom_idx, position)
    mol_out = Chem.RemoveHs(mol_out)
    return mol_out


def mol_to_position_matrix(mol, optimize=False):
    if optimize:
        mol = optimize_mol_position(mol)
    N = mol.GetNumAtoms()
    x = np.zeros([N,3])
    for idx1 in range(N):
        pos = mol.GetConformer().GetAtomPosition(idx1)
        x[idx1,0], x[idx1,1], x[idx1,2] = (pos.x, pos.y, pos.z)
    return x


def mol_to_distance_matrix(mol):
    x_mat = mol_to_position_matrix(mol)
    d = np.zeros([len(x_mat), len(x_mat)])
    for idx1 in range(len(x_mat)):
        for idx2 in range(len(x_mat)):
            d[idx1,idx2] = np.linalg.norm(x_mat[idx1] - x_mat[idx2])
    return d


if __name__ == '__main__':
    m = Chem.MolFromSmiles('CCOC')
    mol_out = optimize_mol_position(m)
    m = mol_to_position_matrix(mol_out)
    print(m)
    d = mol_to_distance_matrix(mol_out)
    print(d)

