import torch
from torch import nn
from torch.nn import functional as F 


class GraphConvolutionLayer(nn.Module):
    """ 
    Class for graph convolution layer
    arg:
        h_dim: input node feature dimension
    input:
        h: node feature matrix [b N nf]
        adj: adjacency matrix [b N N]
    output:
        h: node feature matrix [b N nf]
    """

    def __init__(self, h_dim):
        super().__init__()
        self.fc = nn.Linear(h_dim, h_dim, bias=True)
        
    def forward(self, h, adj):
        h = self.fc(h) # [b N nf]
        h = torch.matmul(adj, h) # [b N nf]
        return h 


class GraphConvolutionBlock(nn.Module):
    """ 
    Class for graph convolution blcok
    arg:
        h_dim: input node feature dimension
        n_layer: number of layers 
        activate_last: activate last (defaul: True)
        residual_connection: method of residual connection
    input:
        h: node feature matrix [b N nf]
        adj: adjacency matrix [b N N]
    output:
        h: node feature matrix [b N nf]
    """

    def __init__(self, h_dim, n_layer=2, activate_last=True, \
            residual_connection='none'):
        super().__init__()
        assert residual_connection in ['none', 'add', 'attention'], \
                'residual connection mode'
        self.n_layer, self.residual_connection = n_layer, residual_connection
        self.activate_last = activate_last
        self.h_dim = h_dim
        self.graph_convolution = nn.ModuleList([GraphConvolutionLayer(h_dim) \
                for _ in range(n_layer)])
        if self.residual_connection == 'attention':
            self.attention_network = nn.Linear(h_dim, 1)

    def forward(self, h, adj):
        b, N = h.size(0), h.size(1)
        
        h_ori = h
        for layer in self.graph_convolution[:-1]:
            h = layer(h, adj)
            h = F.relu(h)
        h = self.graph_convolution[-1](h, adj)

        # residual connection or skip connection method
        if self.residual_connection == 'none':
            pass 
        elif self.residual_connection == 'add':
            h = h + h_ori
        elif self.residual_connection == 'attention':
            h_ori_mean = h_ori.mean(dim=1)
            alpha = self.attention_network(h_ori_mean)
            alpha = torch.sigmoid(alpha).squeeze(1)
            alpha = alpha.unsqueeze(1).unsqueeze(2).repeat(1, N, self.h_dim)
            h = alpha * h + (1 - alpha) * h_ori

        if self.activate_last:
            h = F.relu(h)

        return h
    

class MPNNLayer(nn.Module):
    pass


class MPNNBlock(nn.Module):
    pass


class EGNNLayer(nn.Module):
    pass 


class EGNNBlock(nn.Module):
    pass

    
h = torch.rand(3, 5, 7)
adj = torch.eye(5, 5).unsqueeze(0).repeat(3, 1, 1)
net = GraphConvolutionBlock(7, residual_connection='attention')
h = net(h, adj)
print(h.size())

