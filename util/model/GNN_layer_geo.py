import torch

from torch_geometric.nn import MessagePassing 
from torch_geometric.utils import add_self_loops, degree 


class MPNNLayer(MessagePassing):
    """ 
    Simple MPNN Layer w/o self attention
    """
    def __init__(self, h_dim, e_dim):
        super().__init__(aggr='add')
        self.h_dim, self.e_dim = h_dim, e_dim
        self.fc_m = torch.nn.Linear(h_dim * 2 + e_dim, h_dim)
        self.fc_alpha = torch.nn.Linear(h_dim, 1)

    def forward(self, x, edge_index, edge_attr): # x[N, h_dim]
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x

    def message(self, x_j, x_i, edge_attr):
        m_ij = torch.cat([x_j, x_i, edge_attr], dim=1)
        m_ij = self.fc_m(m_ij) # [e hd]
        return m_ij


class MPNNBlock(torch.nn.Module):
    """ 
    Simple MPNN Block w/o self attention 
    """
    def __init__(self, h_dim, e_dim, n_layer, \
            activate_last=True, residual_connection='none'):
        super().__init__()
        assert residual_connection in ['none', 'add'], 'residual connnection error'
        self.activate_last = activate_last
        self.residual_connection = residual_connection
        self.h_dim, self.e_dim = h_dim, e_dim
        self.layer_list = torch.nn.ModuleList([MPNNLayer(h_dim, e_dim) for _ in range(n_layer)])

    def forward(self, x, edge_index, edge_attr):
        x_ori = x

        for layer in self.layer_list:
            x = layer(x, edge_index, edge_attr)
            x = torch.nn.functional.leaky_relu(x)

        if self.residual_connection == 'none':
            pass 
        elif self.residual_connection == 'add':
            x = x + x_ori

        if self.activate_last:
            x = torch.nn.functional.leaky_relu(x)

        return x


if __name__ == '__main__':
    pass

