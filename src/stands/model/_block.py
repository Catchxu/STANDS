import torch
from torch import nn
import math
from math import sqrt, pi
from torch.nn import functional as F


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
    def __init__(self, d_model, nheads) -> None:
        super().__init__()

        self.d_model = d_model
        self.nheads = nheads
        self.dim = d_model // nheads

        self.proj_q = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_k = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_v = nn.Linear(self.dim, self.dim, bias=False)
        self.fc_out = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, q, k, v):
        N = q.shape[0]

        q = self.proj_q(q.reshape(N, 1, self.nheads, self.dim))
        k = self.proj_k(k.reshape(N, 1, self.nheads, self.dim))
        v = self.proj_v(v.reshape(N, 1, self.nheads, self.dim))

        attn = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        
        attn = F.softmax(attn / (self.d_model ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attn, v]).reshape(
            N, self.d_model
        )

        return self.fc_out(out)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nheads, hidden_dim=1024, dropout=0.3) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(d_model, nheads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v):
        attn = self.attention(q, k, v)
        
        x = self.dropout(self.norm1(attn + q))
        f = self.fc(x)
        x = self.dropout(self.norm2(x + f))
        return x


class TFBlock(nn.Module):
    def __init__(self, d_model, num_layers=3, nheads=4, 
                 hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(d_model*2, nheads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_g, z_p):
        z = self.dropout(torch.concat([z_g, z_p], dim=-1))
        for layer in self.layers:
            z = layer(z, z, z)
        z_g, z_p = torch.chunk(z, 2, dim = -1)
        return z_g, z_p


class StyleBlock(nn.Module):
    def __init__(self, data_n: int, z_dim: int):
        super().__init__()
        self.n = data_n
        self.style = nn.Parameter(torch.Tensor(data_n, z_dim))
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