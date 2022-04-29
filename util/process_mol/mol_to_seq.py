import numpy as np
from rdkit import Chem
import random


a = 'CC(=O)NCCC1=CNc2c1cc(OC)cc2'
# a = 'CN=CO'
m = Chem.MolFromSmiles(a)


def get_all_possible_smiles(m):
    """ 
    get all possible (mostly) smiles for given mol 
    input:
        m: mol object
    output: 
        smiles_list: possible smiles_list 
    """
    smiles_list = []
    while True:
        smiles = Chem.MolToSmiles(m, doRandom=True)
        if not smiles in smiles_list:
            smiles_list.append(smiles)
            c = 0
        else:
            c += 1
        if (c > len(smiles_list) * 10) and (c > 100):
            break
    return smiles_list


def get_random_smiles(mol, kekulize=False, isomeric=False, \
        explicit_bond=False, explicit_H=False, temp=False):
    """ 
    get random smiles (non cannonical) from mol
    """
    if kekulize:
        kekulize = bool(random.randint(0, 1))
    if isomeric:
        isomeric = bool(random.randint(0, 1))
    allHsExplicit, allBondsExplicit = False, False
    if explicit_bond:
        allBondsExplicit = bool(random.randint(0, 1))
        if temp: print('x')
    if explicit_H:
        allHsExplicit = bool(random.randint(0, 1))
        if temp: print('x')

    if True:
        try:
            smiles = Chem.MolToSmiles(mol, doRandom=True, kekuleSmiles=kekulize, \
                    isomericSmiles=isomeric, allBondsExplicit=allBondsExplicit, \
                    allHsExplicit=allHsExplicit)
        except:
            pass

    return smiles


def smiles_to_seq(smiles, c_to_i, make_cannonical=False):
    if cannonical:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    seq = []
    for char in smiles:
        i = c_to_i.index(char)
        seq.append(i)
    seq = np.array(seq)
    return smiles, seq


def update_c_to_i(smiles_list, c_to_i=[], make_cannonical=False):
    if cannonical:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    for smiles in smiles_list:
        for char in smiles:
            if not char in c_to_i:
                c_to_i.append(char)
    return c_to_i


def get_random_smiles(smiles, randint=100):
    
    def dummy_seed(randint):
        for _ in range(random.randint(1, randint)):
            Chem.MolToSmiles(Chem.MolFromSmiles('C'), doRandom=True)
    if True:
        if '.' in smiles:
            smiles_list = smiles.split('.')
        else:
            smiles_list = [smiles]
        mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        new_smiles_list = []
        for mol in mol_list:
            dummy_seed(randint=randint)
            new_smiles_list.append(Chem.MolToSmiles(mol, doRandom=True))

        random.shuffle(new_smiles_list)
        new_smiles = ''
        for smiles in new_smiles_list:
            new_smiles += smiles
            new_smiles += '.'
        new_smiles = new_smiles[:-1]
    return new_smiles


if __name__ == '__main__':
    smiles = a
    for _ in range(10):
        print(get_random_smiles(Chem.MolFromSmiles(smiles), kekulize=True, isomeric=True, explicit_bond=True, explicit_H=True))

