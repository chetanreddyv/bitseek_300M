import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_scale', torch.ones(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight_scale.fill_(1.0)

    def forward(self, x):
        # Quantize weights to 1.58-bit (ternary)
        weight_abs = torch.abs(self.weight)
        weight_mean = weight_abs.mean()
        weight_ternary = torch.where(
            weight_abs > weight_mean,
            torch.sign(self.weight),
            torch.zeros_like(self.weight)
        )
        
        # Quantize activations to 8-bit
        x_abs = torch.abs(x)
        x_max = x_abs.max(dim=-1, keepdim=True)[0]
        x_scale = 127.0 / (x_max + 1e-8)
        x_quantized = torch.round(x * x_scale) / x_scale
        
        # Forward pass with quantized weights and activations
        return F.linear(x_quantized, weight_ternary * self.weight_scale)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
            self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class FlashMLA(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = BitLinear(dim, dim * 3)
        self.proj = BitLinear(dim, dim)
        self.dropout = dropout
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # Check if flash attention is available
        self.use_flash_attn = False
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.use_flash_attn = True
        except ImportError:
            print("Flash attention not available, falling back to standard attention")

    def forward(self, x, mask=None):
        B, L, C = x.shape
        
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.num_heads), qkv)
        
        cos, sin = self.rotary_emb(q, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if self.use_flash_attn:
            # Flash attention implementation
            if mask is not None:
                # Convert attention mask to the format expected by flash attention
                mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
                mask = mask.expand(-1, self.num_heads, -1, -1)  # [B, H, 1, L]
                mask = mask.contiguous()
            
            try:
                out = self.flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=True,  # Use causal attention for language modeling
                    softmax_scale=self.scale,
                    key_padding_mask=mask if mask is not None else None
                )
            except Exception as e:
                print(f"Flash attention failed, falling back to standard attention: {e}")
                out = self._standard_attention(q, k, v, mask)
        else:
            out = self._standard_attention(q, k, v, mask)
        
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.proj(out)
        
        return out
    
    def _standard_attention(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        return torch.matmul(attn, v)

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = BitLinear(in_features, hidden_features)
        self.w2 = BitLinear(in_features, hidden_features)
        self.w3 = BitLinear(hidden_features, in_features)

    def forward(self, x):
        swish = self.w1(x) * torch.sigmoid(self.w2(x))
        return self.w3(swish)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = FlashMLA(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class TinyLM(nn.Module):
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=2048,
        dropout=0.0
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = RotaryEmbedding(hidden_size // num_heads, max_seq_len)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.head = BitLinear(hidden_size, vocab_size, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        B, L = x.shape
        
        # Token embeddings
        x = self.token_embedding(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        logits = self.head(x)
        
        return logits 