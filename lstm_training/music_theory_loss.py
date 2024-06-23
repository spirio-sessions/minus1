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

        # Try to look at all notes at the same time
        """
        harmony_loss = 0
        batch_size, seq_length = harmony.shape
        
        for i in range(batch_size):
            for j in range(seq_length):
                melody_notes = (melodies[i, j] > 0).nonzero(as_tuple=True)
                harmony_notes = (harmony[i, j] > threshold).nonzero(as_tuple=True)

                for melody_note in melody_notes:
                    for harmony_note in harmony_notes:
                        harmony_loss += interval_quality(melody_note, harmony_note)
        """

        # Looking at one note at a time
        melody_notes = melodies.argmax(dim=1)  # Shape: (batch_size, seq_length)
        harmony_notes = harmony.argmax(dim=1)  # Shape: (batch_size, seq_length)

        # Calculate harmony loss using vectorized interval quality
        penalties = interval_quality(melody_notes, harmony_notes)
        harmony_loss = penalties.mean()

        return self.alpha * mse_loss + self.beta * harmony_loss
