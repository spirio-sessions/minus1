import torch
import torch.nn as nn
import math

# Sinusoidal positional embeddings
class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embeddings module.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Calculate sinusoidal positional embeddings
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Transformer block with Attention and causal masking
class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and causal masking.
    """

    def __init__(self, hidden_size=128, num_heads=4):
        super(TransformerBlock, self).__init__()

        # Layer normalization for input
        self.norm1 = nn.LayerNorm(hidden_size)

        # Multi-head self-attention mechanism
        self.multihead_attn = nn.MultiheadAttention(hidden_size,
                                                    num_heads=num_heads,
                                                    batch_first=True,
                                                    dropout=0.1)

        # Layer normalization for attention output
        self.norm2 = nn.LayerNorm(hidden_size)

        # Feedforward neural network
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x, padding_mask):
        # Create causal mask for Attention
        bs, l, h = x.shape
        mask = torch.triu(torch.ones(l, l, device=x.device), 1).bool()

        # Layer normalization
        norm_x = self.norm1(x)

        # Apply multi-head Attention
        x = self.multihead_attn(norm_x, norm_x, norm_x, attn_mask=mask, key_padding_mask=padding_mask)[0] + x

        # Layer normalization
        norm_x = self.norm2(x)

        # Apply feedforward neural network
        x = self.mlp(norm_x) + x
        return x


# "Decoder-Only" Style Transformer with Attention
class Transformer(nn.Module):
    """
    "Decoder-Only" Style Transformer with self-attention.
    """

    def __init__(self, num_emb, hidden_size=128, num_layers=3, num_heads=4):
        super(Transformer, self).__init__()

        # Token embeddings
        # self.embedding = nn.Embedding(num_emb, hidden_size)
        # Use nn.linear for projection instad nn.embedding doesnt work since it adds a dimension
        self.embedding = nn.Linear(num_emb, hidden_size)

        # Positional embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_size)

        # List of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])

        # Output layer
        self.fc_out = nn.Linear(hidden_size, num_emb)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq, pad_token):
        # Mask for padding tokens
        # input_key_mask = input_seq == 0
        # input key mask should be (batch_size, seq_Length)
        # .all(dim=-1) to compare the token in every dimension
        input_key_mask = (input_seq == pad_token).all(dim=-1) # i think i need to set padding token here for mask
        #print(input_key_mask.shape)

        # Embedding input sequence
        input_embs = self.embedding(input_seq)
        #print(input_embs.shape)
        # bs = Batch size
        # l = sequence length
        # h = hidden size
        bs, l, h = input_embs.shape

        # Add positional embeddings to token embeddings
        seq_indx = torch.arange(l, device=input_seq.device)
        pos_emb = self.pos_emb(seq_indx).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_emb


        # Pass through Transformer blocks
        for block in self.blocks:
            embs = block(embs, padding_mask=input_key_mask)

        # Output predictions
        output = self.fc_out(embs)
        return self.sigmoid(output)  # Apply sigmoid to get probabilities (should apply to each feature sperately)