import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(FeedForward, self).__init__()
        hidden_dim = 2 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=1, dropout=0.1):
        super(Attention, self).__init__()
        dim_head = dim
        inner_dim = dim_head * heads
        project_out = not (heads == 1)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # b = attn[0, 0, :, :].cpu().detach().numpy()
        # np.savetxt("test.csv", b)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dropout=0.1):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dropout=dropout),
                FeedForward(dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = self.norm(attn(x) + x)
            x = self.norm(ff(x) + x)
        return x


class DeT(nn.Module):
    def __init__(self, *, seq_len, patch_len, depth, heads, dropout=0.1, emb_dropout=0.1):
        super(DeT, self).__init__()
        assert seq_len % patch_len == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = seq_len // patch_len
        patch_dim = patch_len
        dim = patch_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (p1 p2) -> b p1 p2', p1=num_patches, p2=patch_dim),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dropout)

        # self.to_latent = nn.Identity()

        self.to_seq_embedding = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b p1 p2 -> b (p1 p2)', p1=num_patches, p2=patch_dim),
        )

    def forward(self, seq):  # seq (batch, 512)
        # c = seq.view(1, -1).cpu().detach().numpy()
        # np.savetxt("x.csv", c)

        x = self.to_patch_embedding(seq)
        b, n, _ = x.shape  # x (batch, patch=8, dim=64)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)  # x (batch, patch=8, dim=64)

        # x = self.to_latent(x)

        return self.to_seq_embedding(x)  # seq (batch, 512)

