"""Patch encoder/decoder, pre-norm Transformer core, and the three compared models.

All models share: Linear(8,128) patch encoder + learned positional embeddings, a pre-norm
non-causal Transformer core (width 128, 4 heads, FFN x4, final LayerNorm), and a
Linear(128,8) patch decoder.

  FeedForward(L=2)  : u_hat = D(core_2(E(u0)))                  -- shallow control
  FeedForward(L=24) : u_hat = D(core_24(E(u0)))                 -- untied deep control
  Recursive         : simplified continuous-output TRM, n=2, T=1, K=4, shared core_2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, n_heads, ffn_mult):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, h):
        B, N, D = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(B, N, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        h = h + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return h + self.mlp(self.ln2(h))


class Core(nn.Module):
    """f_theta: a stack of pre-norm blocks followed by a final LayerNorm."""

    def __init__(self, n_layers, d, n_heads, ffn_mult):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, n_heads, ffn_mult) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)

    def forward(self, h):
        for b in self.blocks:
            h = b(h)
        return self.ln_f(h)


class PatchIO(nn.Module):
    def __init__(self, num_patches, patch_size, d):
        super().__init__()
        self.num_patches, self.patch_size = num_patches, patch_size
        self.enc = nn.Linear(patch_size, d)
        self.pos = nn.Parameter(torch.randn(1, num_patches, d) * 0.02)
        self.dec = nn.Linear(d, patch_size)

    def encode(self, u):  # [B, 1024] -> [B, 128, 128]
        return self.enc(u.view(u.shape[0], self.num_patches, self.patch_size)) + self.pos

    def decode(self, h):  # [B, 128, 128] -> [B, 1024]
        return self.dec(h).reshape(h.shape[0], -1)


class FeedForward(nn.Module):
    recursive = False

    def __init__(self, cfg, n_layers):
        super().__init__()
        self.io = PatchIO(cfg["num_patches"], cfg["patch_size"], cfg["d_model"])
        self.core = Core(n_layers, cfg["d_model"], cfg["n_heads"], cfg["ffn_mult"])

    def forward(self, u0):
        return self.io.decode(self.core(self.io.encode(u0)))


class Recursive(nn.Module):
    recursive = True

    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.n_latent, self.n_sup = cfg["n_latent"], cfg["n_sup"]
        self.io = PatchIO(cfg["num_patches"], cfg["patch_size"], d)
        self.core = Core(cfg["core_layers"], d, cfg["n_heads"], cfg["ffn_mult"])
        self.y_init = nn.Parameter(torch.randn(d) * 0.02)
        self.z_init = nn.Parameter(torch.randn(d) * 0.02)

    def init_state(self, u0):
        B, N = u0.shape[0], self.io.num_patches
        return self.y_init.expand(B, N, -1), self.z_init.expand(B, N, -1)

    def refine(self, x, y, z):
        """One supervised refinement step: z <- f(x+y+z) n times, then y <- f(y+z)."""
        for _ in range(self.n_latent):
            z = self.core(x + y + z)
        y = self.core(y + z)
        return y, z

    def forward(self, u0, n_steps=None):
        """Inference: returns the list of predictions after each refinement step."""
        x = self.io.encode(u0)
        y, z = self.init_state(u0)
        preds = []
        for _ in range(n_steps or self.n_sup):
            y, z = self.refine(x, y, z)
            preds.append(self.io.decode(y))
        return preds


def build_model(name, cfg):
    if name == "shallow":
        return FeedForward(cfg, cfg["core_layers"])
    if name == "deep":
        return FeedForward(cfg, cfg["deep_layers"])
    if name == "recursive":
        return Recursive(cfg)
    raise ValueError(name)


def core_layer_calls(name, cfg, n_steps=None):
    """Transformer-layer applications needed for the (final) prediction."""
    if name == "recursive":
        return (n_steps or cfg["n_sup"]) * (cfg["n_latent"] + 1) * cfg["core_layers"]
    return cfg["core_layers"] if name == "shallow" else cfg["deep_layers"]


def flops_per_sample(name, cfg, n_steps=None):
    """Forward-pass matmul FLOPs for one field (2 FLOPs per multiply-add; norms/activations ignored).

    Per Transformer layer on N tokens of width d with FFN ratio m:
      QKV + output projection 4*N*d^2, FFN 2*m*N*d^2, attention scores + weighted sum 2*N^2*d  (MACs).
    """
    N, d, m, P = cfg["num_patches"], cfg["d_model"], cfg["ffn_mult"], cfg["patch_size"]
    layer_macs = (4 + 2 * m) * N * d * d + 2 * N * N * d
    io_macs = 2 * N * P * d  # patch encoder + decoder
    return 2 * (core_layer_calls(name, cfg, n_steps) * layer_macs + io_macs)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
