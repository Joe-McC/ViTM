
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Attention(nn.Module):
    def __init__(
        self, dim=196, num_heads=7, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super(Attention, self).__init__()

        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Layer Normalisation
class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2




class FeedForward(nn.Module):
    """
    Implementation of MLP for transformer
    """

    def __init__(self, dim, hidden_dim, dropout_rate=0.0):
        super(FeedForward, self).__init__()
        """
        Scaled ReLU: https://arxiv.org/pdf/2109.03810.pdf
        """
        """
        Original: https://arxiv.org/pdf/2010.11929.pdf
        """
        self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, dim),
        )


        self._init_weights()

    def _init_weights(self):
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.bias, std=1e-6)

    def forward(self, x):      
        x = self.net(x)

        return x

# Self Guided Attention 
class SGA(nn.Module):
    def __init__(self,hiddenSize,dropout):
        super(SGA, self).__init__()
       
        self.att = Attention(hiddenSize, 7, False, 0.0, 0.0)
        self.FF = FeedForward(hiddenSize, hiddenSize*4, 0.0)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(hiddenSize)
        
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(hiddenSize)
        
    def forward(self, x):
        x = self.norm1(x + self.dropout1(self.att(x)))
        x = self.norm2(x + self.dropout2(self.FF(x)))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.hiddenSize = 196 #512
        self.dropout = 0.1    
        self.layer = 6
        self.depth = 4
        self.imgDim = 14

        self.pos_embedding = nn.Parameter(torch.randn(1, self.hiddenSize + 1, self.hiddenSize))
        self.dropout1 = nn.Dropout(self.dropout)
        
        self.visual_encoders = nn.ModuleList([SGA(self.hiddenSize,self.dropout) for _ in range(self.layer)])

    def forward(self, x):
        x = x.view(1,self.depth,self.imgDim*self.imgDim)
        B,N,C = x.shape
        print(self.pos_embedding.size())
        x += self.pos_embedding[:, :(N)]
        x = self.dropout1(x)
        for visual_enc in self.visual_encoders:
            x = visual_enc(x)
        x = x.view(1,self.depth,self.imgDim,self.imgDim)           
        return x
