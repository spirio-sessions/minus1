import numpy as np
import torch
from torch import Tensor


### ===============
### every inference should have a max input seq
### ===============

def inference(model, context_seq: Tensor, true_seq: Tensor, threshold: float, pad_token: Tensor, max_seq_len: int,
              device):
    generated_logits = []
    binary_seq_with_truth = []

    # Clone and move context sequence to device
    input_seq = context_seq.clone().to(device)
    num_tokens = true_seq.shape[1]

    print("Tokens to generate:", num_tokens)

    with torch.no_grad():
        for i in range(num_tokens):
            # Model prediction for the next token
            pred = model(input_seq, pad_token)
            next_token = pred[:, -1, :]
            generated_logits.append(next_token)
            next_token_prob = torch.sigmoid(next_token)

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

            # Trim the sequence to the max length and keep first token (hopefully sos)
            if input_seq.shape[1] > max_seq_len:
                input_seq = torch.cat((input_seq[:, :1, :], input_seq[:, 2:, :]), dim=1)

    return binary_seq_with_truth, generated_logits


def inference_melody_only(model, context_seq: Tensor, true_seq: Tensor, threshold: float, pad_token: Tensor,
                          max_seq_len: int,
                          device):
    generated_logits = []
    binary_seq_with_truth = []

    # Clone and move context sequence to device
    input_seq = context_seq.clone().to(device)
    num_tokens = true_seq.shape[1]

    print("Tokens to generate:", num_tokens)

    with torch.no_grad():
        for i in range(num_tokens):
            # Model prediction for the next token
            pred = model(input_seq, pad_token)
            next_token = pred[:, -1, :]
            generated_logits.append(next_token)
            next_token_prob = torch.sigmoid(next_token)

            # Apply threshold to obtain binary token
            next_token_bin = (next_token_prob >= threshold).float()

            # Add right-hand part with ground truth
            ground_truth = true_seq[0][i].to(device).unsqueeze(0)

            # only take right hand part from ground truth
            mid_idx = ground_truth.size(1) // 2
            ground_truth = ground_truth[:, mid_idx:]

            # print(f"Next Token shape: {next_token_bin.shape}, Ground truth shape: {ground_truth.shape}")
            assert ground_truth.shape == next_token_bin.shape

            next_token_bin = torch.cat((next_token_bin, ground_truth), dim=-1)

            # Append new Token to input seq
            input_seq = torch.cat((input_seq, next_token_bin.unsqueeze(1)), dim=1)
            binary_seq_with_truth.append(next_token_bin)

            # Trim the sequence to the max length and keep first token (hopefully sos)
            if input_seq.shape[1] > max_seq_len:
                input_seq = torch.cat((input_seq[:, :1, :], input_seq[:, 2:, :]), dim=1)

    return binary_seq_with_truth, generated_logits


def inference_top_k_truth_notes(model, context_seq: Tensor, true_seq: Tensor, pad_token: Tensor, max_seq_len: int,
              device):
    generated_logits = []
    binary_seq_with_truth = []

    # Clone and move context sequence to device
    input_seq = context_seq.clone().to(device)
    num_tokens = true_seq.shape[1]

    print("Tokens to generate:", num_tokens)

    with torch.no_grad():
        for i in range(num_tokens):
            # Model prediction for the next token
            pred = model(input_seq, pad_token)
            next_token = pred[:, -1, :]
            generated_logits.append(next_token)
            next_token_prob = torch.sigmoid(next_token)

            # Calculate the midpoint index that separates left and right hand
            mid_idx = next_token_prob.size(1) // 2

            # Determine k (number of active left-hand notes in the ground truth token)
            ground_truth = true_seq[0][i].to(device).unsqueeze(0)
            k = ground_truth[:, :mid_idx].sum().int().item()

            # Select top k probabilities for the left-hand part
            _, top_k_indices = torch.topk(next_token_prob[:, :mid_idx], k, dim=1)
            next_token_bin = torch.zeros_like(next_token_prob)
            next_token_bin.scatter_(1, top_k_indices, 1.0)

            # Replace right-hand part with ground truth
            next_token_bin[:, mid_idx:] = ground_truth[:, mid_idx:]

            # Append new token to input sequence
            input_seq = torch.cat((input_seq, next_token_bin.unsqueeze(1)), dim=1)
            binary_seq_with_truth.append(next_token_bin)

            # Trim the sequence to the max length and keep the first token (hopefully sos)
            if input_seq.shape[1] > max_seq_len:
                input_seq = torch.cat((input_seq[:, :1, :], input_seq[:, 2:, :]), dim=1)

    return binary_seq_with_truth, generated_logits