import torch
from torch import Tensor

from data_visualization import snapshot_to_midi


def prepare_sequence(sequence: Tensor, context_length: int):
    # blow up the sequence to one batch again if not
    if len(sequence.shape) == 2:
        sequence = torch.unsqueeze(sequence, 0)

    _check_for_sequence_shape(sequence)

    if context_length >= sequence.shape[1]:
        raise ValueError("Context needs to be shorter than whole sequence")

    original_complete_seq = sequence
    context_seq = sequence[:, :context_length]
    continuing_seq = sequence[:, context_length:]

    return context_seq, continuing_seq, original_complete_seq


def _check_for_sequence_shape(sequence: Tensor):
    if len(sequence.shape) != 3:
        raise ValueError(f"Sequence needs to have dimension of 3. Has shape: {sequence.shape}")


def inference_output_to_midi_one_octave(original_complete_seq: Tensor, context_seq: Tensor, last_input_seq: Tensor,
                                        time_per_snapshot: float, save_dir: str, filename: str):
    _check_for_sequence_shape(original_complete_seq)
    _check_for_sequence_shape(context_seq)
    _check_for_sequence_shape(last_input_seq)

    # All sequences need to have batch dim removed
    original_complete_seq = original_complete_seq.squeeze(0)
    context_seq = context_seq.squeeze(0)
    last_input_seq = last_input_seq.squeeze(0)

    track_names = ["Original complete sequence melody",
                   "Original complete sequence harmony",
                   "Context sequence melody",
                   "Context sequence harmony",
                   "Last input seqeunce melody",
                   "Last input seqeunce harmony", ]

    ori_complete_seq_mel, ori_complete_seq_har = snapshot_to_midi.split_snapshots_in_sequence(
        original_complete_seq.cpu().numpy())
    context_seq_mel, context_seq_har = snapshot_to_midi.split_snapshots_in_sequence(context_seq.cpu().numpy())
    last_input_seq_mel, last_input_seq_har = snapshot_to_midi.split_snapshots_in_sequence(
        last_input_seq.cpu().numpy())

    # Blow up each track to 88 keys and move into octave
    ori_complete_seq_mel = snapshot_to_midi.pad_sequence_of_one_hot_vectors(ori_complete_seq_mel, octaves_higher=3)
    ori_complete_seq_har = snapshot_to_midi.pad_sequence_of_one_hot_vectors(ori_complete_seq_har, octaves_higher=1)

    context_seq_mel = snapshot_to_midi.pad_sequence_of_one_hot_vectors(context_seq_mel, octaves_higher=3)
    context_seq_har = snapshot_to_midi.pad_sequence_of_one_hot_vectors(context_seq_har, octaves_higher=1)

    last_input_seq_mel = snapshot_to_midi.pad_sequence_of_one_hot_vectors(last_input_seq_mel, octaves_higher=3)
    last_input_seq_har = snapshot_to_midi.pad_sequence_of_one_hot_vectors(last_input_seq_har, octaves_higher=1)


    # make list of tracks
    tracks = [ori_complete_seq_mel, ori_complete_seq_har, context_seq_mel, context_seq_har, last_input_seq_mel,
              last_input_seq_har]

    # create midi file
    snapshot_to_midi.create_midi_from_snapshots(tracks, track_names, time_per_snapshot,
                                                save_dir, filename)
