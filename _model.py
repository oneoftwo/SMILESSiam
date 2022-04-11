import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LanguageModel(torch.nn.Module):
    """ 
    SMILES seq to latent vector
    """
    def __init__(self, n_char, hid_dim=128, n_layer=2):
        super().__init__()
        
        self.hid_dim, self.n_layer = hid_dim, n_layer
        self.n_char = n_char

        self.embed_seq = nn.Sequential(
                nn.Linear(n_char, hid_dim, bias=False)
                )
        self.bi_gru = nn.GRU(input_size=hid_dim, hidden_size=hid_dim, \
                num_layers=n_layer, dropout=0.3, bidirectional=True, \
                batch_first=True)

        self.fc_head = nn.Sequential(
                nn.Linear(hid_dim * 2, hid_dim),
                nn.BatchNorm1d(hid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hid_dim, hid_dim), 
                nn.BatchNorm1d(hid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hid_dim, hid_dim),
                nn.BatchNorm1d(hid_dim, affine=False)
                )
        
    def forward(self, seq, length):
        seq = F.one_hot(seq, num_classes=self.n_char).float()
        seq = self.embed_seq(seq)
        
        # pack seq
        packed_seq = pack_padded_sequence(seq, length, batch_first=True, \
                enforce_sorted=False)
        output, h = self.bi_gru(packed_seq)
        seq, length = pad_packed_sequence(output, batch_first=True)

        h_forward = h[self.n_layer * 2 - 2]
        h_backward = h[self.n_layer * 2 - 1]

        z = self.fc_head(torch.cat([h_forward, h_backward], dim=1))
        return z


class SMILESSiam(torch.nn.Module):
    def __init__(self, representation_model, use_pp_prediction=False):
        super().__init__()
        
        self.representation_model = representation_model
        hid_dim = representation_model.hid_dim
        self.hid_dim = hid_dim
    
        self.predictor = nn.Sequential(
                nn.Linear(hid_dim, hid_dim * 2),
                nn.BatchNorm1d(hid_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(hid_dim * 2, hid_dim)
                )

        self.use_pp_prediction = use_pp_prediction
        if use_pp_prediction:
            self.fc_pp = nn.Sequential(
                    nn.Linear(hid_dim, hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.Linear(hid_dim, hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.Linear(hid_dim, 20)
                    )
        
    def forward(self, seq_1, length_1, seq_2, length_2):
        
        z1 = self.representation_model(seq_1, length_1)
        z2 = self.representation_model(seq_2, length_2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        if self.use_pp_prediction:
            pp1, pp2 = self.fc_pp(z1), self.fc_pp(z2)
    
        # p1, p2, z1, z2 = p1, p2, z1, z2
        p1, p2, z1, z2 = p1, p2, z1, z2
        
        sample = {'p1': p1, 'p2': p2, 'z1':z1, 'z2':z2}

        if self.use_pp_prediction:
            sample['pp1'] = pp1
            sample['pp2'] = pp2

        return sample
    
    def get_latent(self, seq, length):
        z = self.representation_model(seq, length)
        z = z.detach()
        return z


class SiamClf(torch.nn.Module):
    def __init__(self, siam_model):
        super().__init__()

        self.siam_model = siam_model
        hid_dim = siam_model.hid_dim

        self.fc  = nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.BatchNorm1d(hid_dim),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
                nn.BatchNorm1d(hid_dim),
                nn.Dropout(0.3),
                nn.ReLU(),
                # nn.Linear(hid_dim, hid_dim),
                # nn.BatchNorm1d(hid_dim),
                # nn.Dropout(0.3),
                # nn.ReLU(),
                nn.Linear(hid_dim, 1)
                )

    def forward(self, seq, length):
        z = self.siam_model.get_latent(seq, length)
        r = self.fc(z).squeeze(1)
        return r


if __name__ == '__main__':
    pass 

