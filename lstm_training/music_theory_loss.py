import torch
import torch.nn as nn

# Define a function to compute interval quality
def interval_quality(melody_notes, harmony_notes):
    intervals = torch.abs(melody_notes - harmony_notes) % 12
    penalties = torch.where(intervals == 0, torch.tensor(1.0, device=melody_notes.device),
                            torch.where(intervals == 7, torch.tensor(0.0, device=melody_notes.device),
                            torch.where((intervals == 4) | (intervals == 5), torch.tensor(0.1, device=melody_notes.device),
                            torch.where((intervals == 3) | (intervals == 8), torch.tensor(0.2, device=melody_notes.device),
                            torch.where((intervals == 2) | (intervals == 9), torch.tensor(0.3, device=melody_notes.device),
                            torch.where((intervals == 1) | (intervals == 11), torch.tensor(0.5, device=melody_notes.device),
                            torch.where(intervals == 6, torch.tensor(0.7, device=melody_notes.device), torch.tensor(0.4, device=melody_notes.device))))))))
    return penalties

def interval_quality_all_notes(melody_notes, harmony_notes):
    # Calculate the intervals (in semitones) between each pair of melody and harmony notes
    intervals = torch.abs(melody_notes.unsqueeze(-1) - harmony_notes.unsqueeze(-1).transpose(1, 2)) % 12

    # Define penalties based on the intervals
    # Penalty for unison (0 semitones): 1.0
    penalties = torch.where(intervals == 0, torch.tensor(5.0, device=melody_notes.device),
                # Penalty for perfect fifth (7 semitones): 0.0
                torch.where(intervals == 7, torch.tensor(0.0, device=melody_notes.device),
                # Penalty for major third (4 semitones) or perfect fourth (5 semitones): 0.1
                torch.where((intervals == 4) | (intervals == 5), torch.tensor(0.1, device=melody_notes.device),
                # Penalty for minor third (3 semitones) or minor sixth (8 semitones): 0.2
                torch.where((intervals == 3) | (intervals == 8), torch.tensor(0.2, device=melody_notes.device),
                # Penalty for major second (2 semitones) or major seventh (9 semitones): 0.3
                torch.where((intervals == 2) | (intervals == 9), torch.tensor(0.3, device=melody_notes.device),
                # Penalty for minor second (1 semitone) or minor seventh (11 semitones): 0.5
                torch.where((intervals == 1) | (intervals == 11), torch.tensor(0.5, device=melody_notes.device),
                # Penalty for tritone (6 semitones): 0.7
                torch.where(intervals == 6, torch.tensor(0.7, device=melody_notes.device),
                # Default penalty for other intervals: 0.4
                torch.tensor(0.4, device=melody_notes.device))))))))

    return penalties

# Custom loss function
class MusicTheoryLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(MusicTheoryLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets, melodies, threshold=0.2):
        mse_loss = self.mse_loss(outputs, targets)

        # Ensure tensors are squeezed correctly
        melodies = melodies.squeeze(1)  # Changes melodies Tensor(60, 1, 12) to Tensor(60, 12)

        harmony = outputs.squeeze(1)    # Ensure harmony Tensor is of shape (batch_size, seq_length)

        # Looking at one note at a time
        """
        melody_notes = melodies.argmax(dim=1)  # Shape: (batch_size, seq_length)
        harmony_notes = harmony.argmax(dim=1)  # Shape: (batch_size, seq_length)
        """

        # Looking at all notes at the same time
        # Extracting played notes
        melody_notes = (melodies > 0).nonzero(as_tuple=True)  # Indices of played melody notes
        harmony_notes = (harmony > threshold).nonzero(as_tuple=True)  # Indices of played harmony notes

        # Calculate harmony loss using interval quality
        penalties = interval_quality_all_notes(melodies, harmony)
        num_melody_notes = (melodies > 0).sum(dim=1).float().clamp(min=1)  # Avoid division by zero
        num_harmony_notes = (harmony > threshold).sum(dim=1).float().clamp(min=1)  # Avoid division by zero

        harmony_loss = (penalties.sum(dim=1).sum(dim=1) / (num_melody_notes * num_harmony_notes)).mean()

        return self.alpha * mse_loss + self.beta * harmony_loss
