import torch
from torch import Tensor
from tqdm import tqdm

from transformer_decoder_training.transformer_inference_eval.inference_and_visualize_1 import \
    inference_output_to_midi_one_octave


def prepare_sequence(sequence: Tensor, context_length: int, device):
    sequence = sequence.to(device)
    if len(sequence.shape) == 2:
        sequence = torch.unsqueeze(sequence, 0)
        print("Why did it unsqueeze wtf?")
    if len(sequence.shape) != 3:
        raise ValueError(f"Sequence needs to have dimension of 3. Has shape: {sequence.shape}")
    if context_length >= sequence.shape[1]:
        raise ValueError("Context needs to be shorter than whole sequence")

    # Sequence = (1, seq_length, 24)
    original_complete_seq = sequence
    context_seq = sequence[:, :context_length]
    continuing_seq = sequence[:, context_length:]

    return context_seq, continuing_seq, original_complete_seq


def predict_sequence(model, context_sequence, true_continuing_sequence, device):
    generated_tokens = []
    generated_harmony = []

    # Clone sequence so we don't change the original input
    input_seq = context_sequence.clone().to(device)
    true_continuing_sequence = true_continuing_sequence.to(device)

    print("Tokens to generate:", true_continuing_sequence.shape[1])

    # Initialize the hidden state and cell state for the LSTM
    hidden, cell = model.init_hidden(batch_size=1, device=device)

    with torch.no_grad():
        for i in range(true_continuing_sequence.shape[1]):
            print("iteration:", i)

            # Model prediction for new token
            data_pred, (hidden, cell) = model(input_seq, (hidden, cell))

            # Get last token from output (should be the one new token)
            next_token = data_pred[:, -1, :]

            # Apply sigmoid to tensor since it should be a multi one hot encoded vector
            next_token = torch.sigmoid(next_token)

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

            # Replace the second half with the ground truth vector
            second_half = right_hand_truth

            # Concatenate the first half of the original vector with the ground truth vector
            next_token = torch.cat((first_half, second_half), dim=1)

            # Append the new token to the sequence
            input_seq = torch.cat((input_seq, next_token.unsqueeze(1)), dim=1)

            # Cutting of the last memory above 2048, to avoid memory issues
            if input_seq.shape[1] > 2048:
                input_seq = input_seq[:, 1:]

    return generated_harmony, input_seq


def predict_outcome(model, ground_truth, seq_length: int, device):
    # Get sequence
    sequence = next(iter(ground_truth)).to(device)
    print(sequence.shape)
    context_length = seq_length // 2
    context_seq, continuing_seq, original_seq = prepare_sequence(sequence, context_length, device)
    # first seq_length // 2, second seq_length // 2, whole seq_length

    generated_harmony, last_input_seq = predict_sequence(model, context_seq, continuing_seq, device)

    # Later using original_seq & context_seq & last_input_seq for inference
    # inference_output_to_midi_one_octave(original_seq, context_seq, last_input_seq,
    # 0.05, midi_save_dir, "threshold_only.mid")

    inference_output_to_midi_one_octave(original_seq, context_seq, last_input_seq, 0.025,
                                        'G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\pipeline_lstm_13keys\\04_finished_model', "first_test_midi.mid")

    return generated_harmony, last_input_seq


def predict_entire_song(model, dataloader, context_length: int, device):
    whole_song = next(iter(dataloader)).to(device)
    whole_song = torch.unsqueeze(whole_song, 0)

    context_seq = whole_song[:, :context_length]
    continuing_seq = whole_song[:, context_length:]

    total_generated_harmony = [context_seq.squeeze(0)]
    hidden, cell = model.init_hidden(batch_size=1, device=device)

    max_seq_length = context_length * 2

    for i in tqdm(range(continuing_seq.shape[1]), desc="Processing steps", unit="step", leave=False):
        data_pred, (hidden, cell) = model(context_seq, (hidden, cell))
        next_token = data_pred[:, -1, :]
        next_token = torch.sigmoid(next_token)

        vec_length = next_token.size(1)
        midpoint = vec_length // 2
        ground_truth = whole_song[:, context_length + i].to(device)

        right_hand_truth = ground_truth[:, midpoint:]
        first_half = next_token[:, :midpoint]
        second_half = right_hand_truth

        next_token = torch.cat((first_half, second_half), dim=1)
        total_generated_harmony.append(next_token)
        context_seq = torch.cat((context_seq, next_token.unsqueeze(1)), dim=1)

        if context_seq.shape[1] > max_seq_length:
            context_seq = context_seq[:, -max_seq_length:]

        # Detach hidden and cell states to avoid retaining computation history
        hidden = hidden.detach()
        cell = cell.detach()

    generated_harmony = torch.cat(total_generated_harmony, dim=0).cpu().detach().numpy()

    return generated_harmony
