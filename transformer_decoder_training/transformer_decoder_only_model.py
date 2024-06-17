import torch
import torch.nn as nn

class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim, embed_dim, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.embed_dim = embed_dim

        # Transformer-Decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, nhead, dim_feedforward, dropout),
            num_layers
        )

        # Ausgabeschicht
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def create_positional_encoding(self, length, embed_dim):
        pe = torch.zeros(length, embed_dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_len = src.size(1)
        tgt_len = tgt.size(1)

        src = self.embedding(src) + self.create_positional_encoding(src_len, self.embed_dim).to(src.device)
        tgt = self.embedding(tgt) + self.create_positional_encoding(tgt_len, self.embed_dim).to(tgt.device)

        output = self.transformer_decoder(tgt, src, tgt_mask=tgt_mask, memory_mask=src_mask)
        return self.fc_out(output)
