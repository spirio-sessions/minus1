import json
import torch


def load_transformer_model(params_path: str, model_dict_path, device="cpu"):
    # Transformer without sigmoid output
    from transformer_decoder_training.models.transformer_decoder_2 import Transformer

    with open(params_path, 'r') as file:
        parameters = json.load(file)

    model = Transformer(num_emb=parameters["num_emb"], num_layers=parameters["num_layers"],
                        hidden_size=parameters["hidden_size"], num_heads=parameters["num_heads"]).to(device)
    model.load_state_dict(torch.load(model_dict_path))

    model.eval()