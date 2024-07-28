import numpy as np
import torch
from torch import Tensor

### ===============
### every inference should have a max input seq
### ===============

def inference(model, context_seq: Tensor, true_seq: Tensor, threshold: float, pad_token: Tensor, max_seq_len: int, device):
    generated_probs = []
    binary_seq_with_truth = []

    # Clone and move context sequence to device
    input_seq = context_seq.clone().to(device)
    num_tokens = true_seq.shape[1]

    print("Tokens to generate:", num_tokens)

    with torch.no_grad():
        for i in range(num_tokens):
            # Model prediction for the next token
            pred = model(input_seq, pad_token)
            next_token_prob = torch.sigmoid(pred[:, -1, :])
            generated_probs.append(next_token_prob)

            # Apply threshold to obtain binary token
            next_token_bin = (next_token_prob >= threshold).float()
            mid_idx = next_token_bin.size(1) // 2

            # Replace right-hand part with ground truth
            ground_truth = true_seq[0][i].to(device).unsqueeze(0)
            assert ground_truth.shape == next_token_bin.shape

            next_token_bin[:, mid_idx:] = ground_truth[:, mid_idx:]

            # Append new Token to input seq
            input_seq = torch.cat((input_seq, next_token_bin.unsqueeze(1)), dim=1)
            binary_seq_with_truth.append(next_token_bin)

            # Trim the sequence to the max length
            if input_seq.shape[1] > max_seq_len:
                input_seq = torch.cat((input_seq[:, :1, :], input_seq[:, 2:, :]), dim=1)

    return binary_seq_with_truth, generated_probs
