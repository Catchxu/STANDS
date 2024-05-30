import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATv2Conv


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, nhead, norm: bool = True,
                 act: bool = True, dropout: bool = True):
        super().__init__()
        self.layer = GATv2Conv(in_dim, out_dim, num_heads=nhead)
        self.fc = nn.Sequential(
            nn.Linear(nhead*out_dim, out_dim),
            nn.BatchNorm1d(out_dim) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
        )

    def forward(self, g, feat):
        feat = self.layer(g, feat)
        z = self.fc(feat.flatten(1))
        return z




class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm: bool = True,
                 act: bool = True, dropout: bool = True):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
        )

    def forward(self, x):
        x = self.linear(x)
        return x




class MemoryBlock(nn.Module):
    def __init__(self, mem_dim, z_dim, shrink_thres=0.005, tem=0.5):
        super().__init__()
        self.mem_dim = mem_dim
        self.z_dim = z_dim
        self.shrink_thres = shrink_thres
        self.tem = tem
        self.register_buffer("mem", torch.randn(self.mem_dim, self.z_dim))
        self.register_buffer("mem_ptr", torch.zeros(1, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mem.size(1))
        self.mem.data.uniform_(-stdv, stdv)

    @torch.no_grad()
    def update_mem(self, z):
        batch_size = z.shape[0]  # z, B x C
        ptr = int(self.mem_ptr)
        assert self.mem_dim % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.mem[ptr:ptr + batch_size, :] = z  # mem, M x C
        ptr = (ptr + batch_size) % self.mem_dim  # move pointer

        self.mem_ptr[0] = ptr

    def hard_shrink_relu(self, x, lambd=0, epsilon=1e-12):
        x = (F.relu(x-lambd) * x) / (torch.abs(x - lambd) + epsilon)
        return x

    def forward(self, x):
        att_weight = torch.mm(x, self.mem.T)
        att_weight = F.softmax(att_weight/self.tem, dim=1)

        # ReLU based shrinkage, hard shrinkage for positive value
        if (self.shrink_thres > 0):
            att_weight = self.hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        output = torch.mm(att_weight, self.mem)
        return output




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nheads, dropout=0.1) -> None:
        super().__init__()
        assert d_model % nheads == 0

        self.d_k = d_model // nheads
        self.h = nheads
        self.dropout = dropout

        # Produce N identical layers
        self.linears = nn.ModuleList([
            copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)
        ])
    
    def attention(query, key, value, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            attn = dropout(attn)
        return torch.matmul(attn, value), attn        

    def forward(self, q, k, v):
        N = q.shape[0]

        q, k, v = [
            l(x).view(N, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (q, k, v))
        ]

        x, self.attn = self.attention(q, k, v, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(N, self.h*self.d_k)

        return self.linears[-1](x)




class TransformerLayer(nn.Module):
    def __init__(self, d_model, nheads, hidden_dim=1024, dropout=0.3) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(d_model, nheads)

        self.norm = nn.ModuleList([
            copy.deepcopy(nn.LayerNorm(d_model)) for _ in range(2)
        ])

        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v):
        attn = self.attention(q, k, v)

        x = self.dropout(self.norm[0](attn + q))
        f = self.fc(x)
        x = self.dropout(self.norm[1](x + f))
        return x




class TFBlock(nn.Module):
    def __init__(self, p_dim, g_dim, num_layers=3, nheads=4, 
                 hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(p_dim + g_dim, nheads, hidden_dim, dropout) 
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, z_g, z_p):
        z = self.dropout(torch.concat([z_g, z_p], dim=-1))
        for layer in self.layers:
            z = layer(z, z, z)
        z_g, z_p = torch.chunk(z, 2, dim = -1)
        return z_g, z_p




class CrossTFBlock(nn.Module):
    def __init__(self, p_dim, g_dim, num_layers=3, nheads=4, 
                 hidden_dim=1024, dropout=0.1):
        super().__init__()
        dim = min(p_dim, g_dim)
        if p_dim != g_dim:
            self.linear_p = nn.Linear(p_dim, dim)
            self.linear_g = nn.Linear(g_dim, dim)
            self.map_flag = True

        self.p_layers = nn.ModuleList([
            TransformerLayer(dim, nheads, hidden_dim, dropout) 
            for _ in range(num_layers)
        ])

        self.g_layers = nn.ModuleList([
            TransformerLayer(dim, nheads, hidden_dim, dropout) 
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z_g, z_p):
        if self.map_flag:
            z_g = self.dropout(self.linear_g(z_g))
            z_p = self.dropout(self.linear_p(z_p))
        else:
            z_g = self.dropout(z_g)
            z_p = self.dropout(z_p)

        for g_layer, p_layer in zip(self.g_layers, self.p_layers):
            z_g = g_layer(z_g, z_p, z_p)
            z_p = p_layer(z_p, z_g, z_g)
        
        return z_g, z_p




class StyleBlock(nn.Module):
    def __init__(self, n_batch, z_dim):
        super().__init__()
        self.n = n_batch
        self.style = nn.Parameter(torch.Tensor(n_batch, z_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.style.size(1))
        self.style.data.uniform_(-stdv, stdv)

    def forward(self, z, batchid):
        if self.n == 1:
            return z - self.style
        else:
            s = torch.mm(batchid, self.style)
            return z - s