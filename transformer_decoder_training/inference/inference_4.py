# inference with sigmoid in inference and temperature sampling
import numpy as np
import torch


def sample_with_temperature(logits, temperature):
    probabilities = torch.sigmoid(logits / temperature)  # Apply temperature to logits
    return torch.bernoulli(probabilities)  # Sample from the probabilities


def inference_with_temperature_sampling(model, context_sequence, true_continuing_sequence, temperature, pad_token,
                                        device):
    generated_tokens_probabilities = []
    generated_harmony = []

    # Clone sequence so we don't change the original input
    input_seq = context_sequence.clone()

    print("Tokens to generate:", true_continuing_sequence.shape[1])

    with torch.no_grad():
        for i in range(true_continuing_sequence.shape[1]):

            input_seq = input_seq.to(device)

            # Model prediction for new token
            data_pred = model(input_seq, pad_token)

            if data_pred.shape[1] > input_seq.shape[1]:
                print("Model might have generated more than one token")

            # Get last token from output (should be the one new token)
            next_token = data_pred[:, -1, :]

            # Add token to list
            generated_tokens_probabilities.append(next_token)

            # Apply temperature sampling to the logits
            next_token = sample_with_temperature(next_token, temperature)
            print("Token after temperature sampling: ", next_token)

            # Replace generated right hand with right hand ground truth
            # Determine the midpoint of the vector
            vec_length = next_token.size(1)
            midpoint = vec_length // 2

            # Get the ground truth right hand:
            ground_truth = true_continuing_sequence[0][i].to(device)
            ground_truth = torch.unsqueeze(ground_truth, 0)

            assert ground_truth.shape == next_token.shape  # Ensure same dimensions

            # Get right hand truth
            right_hand_truth = ground_truth[:, midpoint:]

            # Split the original vector into two halves
            first_half = next_token[:, :midpoint]
            generated_harmony.append(first_half)

            # Replace the second half with the ground truth vector
            second_half = right_hand_truth

            # Concatenate the first half of the original vector with the ground truth vector
            next_token = torch.cat((first_half, second_half), dim=1)

            # Append the new token to the sequence
            input_seq = torch.cat((input_seq, next_token.unsqueeze(1)), dim=1)

    return generated_tokens_probabilities, generated_harmony, input_seq


import torch


def sample_with_temperature_and_max_notes(logits, temperature):
    probabilities = torch.sigmoid(logits / temperature)  # Apply temperature to logits
    return probabilities


def inference_with_temperature_and_max_notes_sampling(model, context_sequence, true_continuing_sequence, threshold,
                                                      temperature, pad_token, device, max_notes_per_time_step):
    generated_tokens_probabilities = []
    generated_harmony = []

    # Clone sequence so we don't change the original input
    input_seq = context_sequence.clone()

    print("Tokens to generate:", true_continuing_sequence.shape[1])

    with torch.no_grad():
        for i in range(true_continuing_sequence.shape[1]):

            input_seq = input_seq.to(device)

            # Model prediction for new token
            data_pred = model(input_seq, pad_token)

            if data_pred.shape[1] > input_seq.shape[1]:
                print("Model might have generated more than one token")

            # Get last token from output (should be the one new token)
            next_token = data_pred[:, -1, :]

            # Add token to list
            generated_tokens_probabilities.append(binary_next_token)

            # Apply temperature sampling to the logits
            next_token_probs = sample_with_temperature(next_token, temperature)
            print("Token probabilities after temperature sampling: ", next_token_probs)

            # Convert probabilities to binary values with threshold and max notes constraint
            sorted_indices = torch.argsort(next_token_probs, descending=True)
            binary_next_token = torch.zeros_like(next_token_probs)

            for idx in sorted_indices[0, :max_notes_per_time_step]:
                if next_token_probs[0, idx] >= threshold:
                    binary_next_token[0, idx] = 1.0

            print("Binary token after applying threshold and max notes constraint: ", binary_next_token)

            # Replace generated right hand with right hand ground truth
            # Determine the midpoint of the vector
            vec_length = binary_next_token.size(1)
            midpoint = vec_length // 2

            # Get the ground truth right hand:
            ground_truth = true_continuing_sequence[0][i].to(device)
            ground_truth = torch.unsqueeze(ground_truth, 0)

            assert ground_truth.shape == binary_next_token.shape  # Ensure same dimensions

            # Get right hand truth
            right_hand_truth = ground_truth[:, midpoint:]

            # Split the original vector into two halves
            first_half = binary_next_token[:, :midpoint]
            generated_harmony.append(first_half)

            # Replace the second half with the ground truth vector
            second_half = right_hand_truth

            # Concatenate the first half of the original vector with the ground truth vector
            next_token = torch.cat((first_half, second_half), dim=1)

            # Append the new token to the sequence
            input_seq = torch.cat((input_seq, next_token.unsqueeze(1)), dim=1)

    return generated_tokens_probabilities, generated_harmony, input_seq


def inference_with_max_input_sequence(model, context_sequence: torch.Tensor, true_continuing_sequence: torch.Tensor,
                                      threshold: float, pad_token: torch.Tensor, max_input_sequence_length: int,
                                      device):
    # TODO: deprecate using last input sequence as output in general
    generated_tokens_probabilities = []
    generated_harmony = []

    # Clone sequence so we don't change the original input
    # input sequence has shape: (batch_size, seq_length, snapshot_dim)
    input_seq = context_sequence.clone()

    print("Tokens to generate:", true_continuing_sequence.shape[1])

    with torch.no_grad():
        for i in range(true_continuing_sequence.shape[1]):

            # print("iteration:", i)

            input_seq = input_seq.to(device)

            # print("input sequence shape:", input_seq.shape)
            # Model prediction for new token
            data_pred = model(input_seq, pad_token)

            if data_pred.shape[1] > input_seq.shape[1]:
                print("Model might have generated more than one token")

            # Get last token from output (should be the one new token)
            next_token = data_pred[:, -1, :]
            # print("Next token shape:", next_token.shape)

            # Apply sigmoid to tensor since it should be a multi one hot encoded vector
            next_token = torch.sigmoid(next_token)
            print("Token after sigmoid: ", next_token)

            # Add token to list
            generated_tokens_probabilities.append(next_token)

            # Change probability values to binary with threshold
            next_token = (next_token >= threshold).float()
            # print("Next token before splitting:", next_token)

            # Replace generated right hand with right hand ground truth
            # Determine the midpoint of the vector
            vec_length = next_token.size(1)  # Should be size(1) for the correct dimension
            midpoint = vec_length // 2

            # Get the ground truth right hand:
            ground_truth = true_continuing_sequence[0][i].to(device)
            ground_truth = torch.unsqueeze(ground_truth, 0)
            # print("Next token shape", next_token.shape)
            # print("ground truth shape", ground_truth.shape)

            assert ground_truth.shape == next_token.shape  # Ensure same dimensions

            # Get right hand truth
            right_hand_truth = ground_truth[:, midpoint:]

            # Split the original vector into two halves
            first_half = next_token[:, :midpoint]
            generated_harmony.append(first_half)
            # print("first half:", first_half)

            # Replace the second half with the ground truth vector
            second_half = right_hand_truth

            # Concatenate the first half of the original vector with the ground truth vector
            next_token = torch.cat((first_half, second_half), dim=1)
            # print("Token with ground truth:", next_token)

            # Append the new token to the sequence
            input_seq = torch.cat((input_seq, next_token.unsqueeze(1)), dim=1)
            # print("last input seq snapshot after adding next token:", input_seq[0][-1])
            # print("input seq shape:", input_seq.shape)

            # Limit the input sequence to specified length
            if input_seq.shape[1] > max_input_sequence_length:
                print("limit the input sequence with shape: ", input_seq.shape)

                # leave the first token since it should be the sos token
                # -> Should be at first index in every input sequence
                # remove token at index 1
                input_seq = torch.cat((input_seq[:, :1, :], input_seq[:, 2:, :]), dim=1)
                print("Shape after deleting old token: ", input_seq.shape)

    return generated_tokens_probabilities, generated_harmony, input_seq
