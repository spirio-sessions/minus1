# Inference for Model without sigmoid output

import torch

def inference(model, context_sequence, true_continuing_sequence, threshold, pad_token, device):
    generated_tokens_probabilities = []
    generated_harmony = []

    # Clone sequence so we don't change the original input
    input_seq = context_sequence.clone()

    print("Tokens to generate:", true_continuing_sequence.shape[1])

    with torch.no_grad():
        for i in range(true_continuing_sequence.shape[1]):

            #print("iteration:", i)

            input_seq = input_seq.to(device)

            #print("input sequence shape:", input_seq.shape)
            # Model prediction for new token
            data_pred = model(input_seq, pad_token)

            if data_pred.shape[1] > input_seq.shape[1]:
                print("Model might have generated more than one token")

            # Get last token from output (should be the one new token)
            next_token = data_pred[:, -1, :]
            #print("Next token shape:", next_token.shape)

            # Apply sigmoid to tensor since it should be a multi one hot encoded vector
            next_token = torch.sigmoid(next_token)
            print("Token after sigmoid: ", next_token)

            # Add token to list
            generated_tokens_probabilities.append(next_token)

            # Change probability values to binary with threshold
            next_token = (next_token >= threshold).float()
            #print("Next token before splitting:", next_token)

            # Replace generated right hand with right hand ground truth
            # Determine the midpoint of the vector
            vec_length = next_token.size(1)  # Should be size(1) for the correct dimension
            midpoint = vec_length // 2

            # Get the ground truth right hand:
            ground_truth = true_continuing_sequence[0][i].to(device)
            ground_truth = torch.unsqueeze(ground_truth, 0)
            #print("Next token shape", next_token.shape)
            #print("ground truth shape", ground_truth.shape)

            assert ground_truth.shape == next_token.shape  # Ensure same dimensions

            # Get right hand truth
            right_hand_truth = ground_truth[:, midpoint:]

            # Split the original vector into two halves
            first_half = next_token[:, :midpoint]
            generated_harmony.append(first_half)
            #print("first half:", first_half)

            # Replace the second half with the ground truth vector
            second_half = right_hand_truth

            # Concatenate the first half of the original vector with the ground truth vector
            next_token = torch.cat((first_half, second_half), dim=1)
            #print("Token with ground truth:", next_token)

            # Append the new token to the sequence
            input_seq = torch.cat((input_seq, next_token.unsqueeze(1)), dim=1)
            #print("last input seq snapshot after adding next token:", input_seq[0][-1])
            #print("input seq shape:", input_seq.shape)

    return generated_tokens_probabilities, generated_harmony, input_seq