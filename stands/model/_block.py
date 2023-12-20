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
    def __init__(self, d_model, num_heads, attention_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.dropout = nn.Dropout(attention_dropout)
        self.norm = nn.LayerNorm(d_model)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        return x.view(batch_size, self.num_heads, self.head_dim)

    def cos_score(self, q, k):
        norm_q = torch.norm(q, dim=-1, keepdim=True)
        norm_k = torch.norm(k, dim=-1, keepdim=True)
        q_normalized = q / norm_q
        k_normalized = k / norm_k

        dot_product = torch.matmul(q_normalized, k_normalized.transpose(-1, -2))
        scale = q.size(-1)
        cosine_similarity = dot_product / scale
        return cosine_similarity

    def forward(self, q, k, v):
        batch_size = q.size(0)

        out_q = q = self.query_projection(q)
        k = self.key_projection(k)
        v = self.value_projection(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        s = self.dropout(self.cos_score(q, k))
        out = torch.matmul(s, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, self.num_heads * self.head_dim)
        out = self.out_projection(out)

        return self.norm(out_q + out)


class TFBlock(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.attn = MultiHeadAttention(d_model*2, num_heads)

    def forward(self, z_g, z_p):
        z = torch.concat([z_g, z_p], dim=-1)
        z = self.attn(z, z, z)
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