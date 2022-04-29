import pickle 

fn = './PubChem/PubChem_1000000.pkl'
a = pickle.load(open(fn, 'rb'))

# p = len([x for x in a if x['target'] == 1])
# n = len([x for x in a if x['target'] == 0])

# print(p, n)

# print(n / p)


for x in a:
    if 'Q' in x['smiles']:
        print(x)
    if 'q' in x['smiles']:
        print(x)
    if 'X' in x['smiles']:
        print(x)
