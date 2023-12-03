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
        # filter_value = -float('Inf')
        # indices_to_remove = att_weight < torch.topk(att_weight, k=20, dim=-1)[0][..., -1, None]
        # att_weight[indices_to_remove] = filter_value
        att_weight = F.softmax(att_weight/self.tem, dim=1)

        # ReLU based shrinkage, hard shrinkage for positive value
        if (self.shrink_thres > 0):
            att_weight = self.hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        output = torch.mm(att_weight, self.mem)
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
    
    def cos_score(self, q, k):
        norm_q = torch.norm(q, dim=1).unsqueeze(-1)
        norm_k = torch.norm(k, dim=1).unsqueeze(-1)
        return torch.matmul(q, k.transpose(-1,1)) / (norm_q * norm_k)

    def forward(self, q, k, v):
        q = self.query_projection(q).unsqueeze(-1)
        k = self.key_projection(k).unsqueeze(-1)
        v = self.value_projection(v).unsqueeze(-1)
        s = self.dropout(self.cos_score(q, k))
        out = torch.matmul(s, v).squeeze(-1)
        return self.norm(q.squeeze(-1) + self.out_projection(out))


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