import torch
from torch import nn
import math
from math import sqrt, pi
from torch.nn import functional as F
from .._utils import hard_shrink_relu


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
        ###VQ Codebook
        self.beta = 0.25
        self.batch_size = 128
        #self.fc = nn.Linear(self.mem_dim, self.z_dim)
        self.embedding = nn.Embedding(self.batch_size, self.z_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.batch_size, 1.0 / self.batch_size)

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

    def forward(self, input):
        att_weight = torch.mm(input, self.mem.T)  # input x mem^T, (BxC) x (CxM) = B x M
        att_weight = F.softmax(att_weight/self.tem, dim=1)  # B x M

        # ReLU based shrinkage, hard shrinkage for positive value
        if (self.shrink_thres > 0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        output = torch.mm(att_weight, self.mem)  # AttWeight x mem, (BxM) x (MxC) = B x C
        return output


class Attention(nn.Module):
    def __init__(self, d_model, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.norm = nn.LayerNorm(d_model)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        q = self.query_projection(q)
        k = self.key_projection(k)
        v = self.value_projection(v)

        s = self.dropout(torch.cosine_similarity(q, k, eps=1e-08))
        out = torch.mul(s.reshape(-1, 1), v)
        return self.norm(q + self.out_projection(out))


class TFBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.g_attn = Attention(d_model=d_model)
        self.p_attn = Attention(d_model=d_model)

    def forward(self, z_g, z_p):
        z_g = self.g_attn(z_g, z_p, z_p)
        z_p = self.p_attn(z_p, z_g, z_g)
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