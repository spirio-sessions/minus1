import torch
import torch.nn as nn

# Usage: criterion = MusicTheoryLoss(alpha_loss, beta_loss) # Alpha equals weight of MSE, beta weight of custom loss-function
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
    penalties = torch.where(intervals == 0, torch.tensor(2.0, device=melody_notes.device),
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

def interval_quality_based_on_highest(melody_notes, harmony_notes):
    # Find the highest note in the melody
    highest_note = melody_notes.max(dim=1, keepdim=True).values

    # Calculate intervals between the highest melody note and all harmony notes
    intervals = torch.abs(highest_note - harmony_notes) % 12

    # Define penalties based on the intervals
    penalties = torch.where(intervals == 0, torch.tensor(1.0, device=melody_notes.device),
                torch.where(intervals == 7, torch.tensor(0.0, device=melody_notes.device),
                torch.where((intervals == 4) | (intervals == 5), torch.tensor(0.1, device=melody_notes.device),
                torch.where((intervals == 3) | (intervals == 8), torch.tensor(0.2, device=melody_notes.device),
                torch.where((intervals == 2) | (intervals == 9), torch.tensor(0.3, device=melody_notes.device),
                torch.where((intervals == 1) | (intervals == 11), torch.tensor(0.5, device=melody_notes.device),
                torch.where(intervals == 6, torch.tensor(0.7, device=melody_notes.device), torch.tensor(0.4, device=melody_notes.device))))))))

    return penalties

# Custom loss function
class MusicTheoryLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(MusicTheoryLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nn_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets, threshold=0.2):
        """
        :param outputs: predictions made by model
        :param targets: ground truth values corresponding to the "outputs"
        :param threshold: threshhold at what point a note gets played or not
        :return: a double-value that represents the amount of loss calculated
        """
        # Separate harmony and melody components from outputs and targets
        harmony_output = outputs[:, :12]
        melody_output = outputs[:, 12:]
        harmony_target = targets[:, :12]
        melody_target = targets[:, 12:]

        bce_loss = self.nn_loss(outputs, targets)

        penalties = interval_quality_based_on_highest(melody_target, harmony_output)

        num_melody_notes = (melody_target > 0).sum(dim=1).float().clamp(min=1)
        num_harmony_notes = (harmony_output > threshold).sum(dim=1).float().clamp(min=1)

        harmony_loss = (penalties.sum(dim=1) / (num_melody_notes * num_harmony_notes)).mean()

        return self.alpha * bce_loss + self.beta * harmony_loss