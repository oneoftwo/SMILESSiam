import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs

from . import mol_to_graph as MOLTOGRAPH 
from . import mol_to_3d as MOLTO3D


def mol_to_edge_index_and_edge_attr(mol): 
    # TODO: accelerate
    adj = MOLTOGRAPH.mol_to_adjacency_matrix(mol, is_self_loop=False)
    e = MOLTOGRAPH.mol_to_edge_feature_matrix(mol, include_extra=False)
    N = adj.shape[0]
    ei_1 = []
    ei_2 = []
    edge_attr = []
    for i in range(N):
        for j in range(N):
            if adj[i, j] == 1:
                edge_feature = e[i, j]
                ei_1.append(i)
                ei_2.append(j)
                edge_attr.append(list(edge_feature))
    edge_index = torch.Tensor([ei_1, ei_2]).long()
    edge_attr = torch.Tensor(edge_attr)
    return edge_index, edge_attr


def mol_to_pos(mol):
    x = MOLTO3D.mol_to_position_matrix(mol, optimize=True)
    x = torch.Tensor(x)
    return x


def mol_to_data(mol, pos=False):
    mol = RemoveHs(mol)
    x = MOLTOGRAPH.mol_to_node_feature_matrix(mol)
    x = torch.Tensor(x)
    edge_index, edge_attr = mol_to_edge_index_and_edge_attr(mol)
    if pos:
        pos = mol_to_pos(mol)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    else:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


if __name__ == '__main__':
    mol = Chem.MolFromSmiles('CNCC')
    data = mol_to_data(mol)
    print(data)

