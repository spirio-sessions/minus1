import torch.nn as nn


# Define a function to compute interval quality
def interval_quality(melody_note, harmony_note):
    interval = abs(melody_note - harmony_note) % 12

    if interval == 0:  # Octave or unison
        return 1.0  # High penalty for simple octaves
    elif interval == 7:  # Perfect unison and perfect fifth
        return 0.0  # No penalty
    elif interval in [4, 5]:  # Major third and perfect fourth
        return 0.1
    elif interval in [3, 8]:  # Minor third and minor sixth
        return 0.2
    elif interval in [2, 9]:  # Major second and major sixth
        return 0.3
    elif interval in [1, 11]:  # Minor second and major seventh
        return 0.5
    elif interval in [6]:  # Tritone
        return 0.7
    else:
        return 0.4


# Custom loss function
class MusicTheoryLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(MusicTheoryLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets, melodies, threshold=0.2):

        mse_loss = self.mse_loss(outputs, targets)

        melodies = melodies.squeeze(1)  # Changes melodies Tensor(60, 1, 12) to Tensor(60, 12)
        harmony = outputs.squeeze(1)

        harmony_loss = 0
        batch_size, seq_length = harmony.shape

        # Try to look at all notes at the same time
        """
        for i in range(batch_size):
            for j in range(seq_length):
                melody_notes = (melodies[i, j] > 0).nonzero(as_tuple=True)
                harmony_notes = (harmony[i, j] > threshold).nonzero(as_tuple=True)

                for melody_note in melody_notes:
                    for harmony_note in harmony_notes:
                        harmony_loss += interval_quality(melody_note, harmony_note)
        """

        # Only looking at one note at a time
        for i in range(batch_size):
            for j in range(seq_length):
                melody_note = melodies[i, j].argmax().item()
                harmony_note = harmony[i, j].argmax().item()
                harmony_loss += interval_quality(melody_note, harmony_note)

        harmony_loss /= batch_size * seq_length

        return self.alpha * mse_loss + self.beta * harmony_loss
