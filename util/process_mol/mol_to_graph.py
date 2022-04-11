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


def _get_atom_feature(atom):
    symbol = _one_of_k_encoding_unk(atom.GetSymbol(), \
            ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'ELSE'])

    degree = _one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 'ELSE'])

    num_h = _one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 'ELSE'])

    valance = _one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 'ELSE'])

    formal_charge = _one_of_k_encoding_unk(atom.GetFormalCharge(), \
            [-2, -1, 0, 1, 2, 3, 'ELSE'])

    hybrdiation = _one_of_k_encoding_unk(atom.GetHybridization(), \
            [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, \
            Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, \
            Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, \
            Chem.rdchem.HybridizationType.UNSPECIFIED])

    chiral = _one_of_k_encoding_unk(atom.GetChiralTag(), \
            [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, \
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, \
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, \
            Chem.rdchem.ChiralType.CHI_OTHER])

    rdfing_size = [atom.IsInRingSize(i) for i in range(3, 9)]

    is_aromatic = [atom.GetIsAromatic()]

    output = []
    output += symbol # 9
    output += num_h 
    output += valance 
    output += formal_charge 
    output += hybrdiation 
    output += chiral 
    output += rdfing_size 
    output += is_aromatic

    return output # total 67 degree


# make one of k encoding block
def _one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# mol to edge feature matrix
def mol_to_edge_feature_matrix(mol, include_extra=True):
    n_bond_features = 5
    if include_extra:
        n_extra_bond_features = 6
    else:
        n_extra_bond_features = 0
    n_atoms = mol.GetNumAtoms()
    E = np.zeros((n_atoms, n_atoms, n_bond_features + n_extra_bond_features))
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)  # rdkit.Chem.Atom
        for j in range(n_atoms):
            e_ij = mol.GetBondBetweenAtoms(i, j)  # rdkit.Chem.Bond
            if e_ij != None:
                e_ij = _get_bond_feature(e_ij, include_extra) \
                        # ADDED edge feat; one-hot vector
                e_ij = list(map(lambda x: 1 if x == True else 0, e_ij)) \
                        # ADDED edge feat; one-hot vector
                E[i,j,:] = np.array(e_ij)
    return E # total 11 degree


def _get_bond_feature(bond, include_extra=False):
    bt = bond.GetBondType()  # rdkit.Chem.BondType
    retval = [
      bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
      bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
      0 # no bond
      #bond.GetIsConjugated(),
      #bond.IsInRing()
      ]
    if include_extra:
        bs = bond.GetStereo()
        retval += [bs == Chem.rdchem.BondStereo.STEREONONE,
                   bs == Chem.rdchem.BondStereo.STEREOANY,
                   bs == Chem.rdchem.BondStereo.STEREOZ,
                   bs == Chem.rdchem.BondStereo.STEREOE,
                   bs == Chem.rdchem.BondStereo.STEREOCIS,
                   bs == Chem.rdchem.BondStereo.STEREOTRANS]
    return np.array(retval)


def mol_to_node_feature_matrix(mol):
    h = []
    for atom in mol.GetAtoms():
        atom_feature = _get_atom_feature(atom)
        h.append(atom_feature)
    h = np.array(h).astype(float)
    return h


def mol_to_adjacency_matrix(mol, is_self_loop=False):
    adj = GetAdjacencyMatrix(mol)
    if is_self_loop:
        for i in range(len(adj)):
            adj[i, i] = 1
    adj = adj.astype(float)
    return adj


if __name__ == '__main__':
    from rdkit import Chem
    mol = Chem.MolFromSmiles('CNF')
    a = mol_to_node_feature_matrix(mol)
    print(a.shape)
    pass 

