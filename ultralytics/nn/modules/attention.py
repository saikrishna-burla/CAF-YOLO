import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True, unbiased=False) + self.eps).sqrt() * self.weight + self.bias

class MSFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.norm = LayerNorm(dim)
        self.pw1 = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dw = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.pw2(x)
        return x

class CAFMAttention(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.conv_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv_v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        attn = torch.sigmoid(self.pw_conv(self.dw_conv(q * k)))
        x = v * attn
        x = self.proj(x)
        return x

