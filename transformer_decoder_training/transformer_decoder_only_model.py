import torch
import torch.nn as nn


class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim, embed_dim, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        # Einbettungsschicht
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Positionscodierung
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, embed_dim))  # 1000 ist die maximale Sequenzlänge

        # Transformer-Decoder
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, nhead, dim_feedforward, dropout),
            num_layers
        )

        # Ausgabeschicht
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, src, tgt):
        # Einbetten und Positionscodierung hinzufügen
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        # Transformer-Decoder
        output = self.transformer_decoder(tgt, src)

        # Ausgabeschicht
        return self.fc_out(output)
