import torch
import numpy as np


def autoregressive_process(model, input_sequence, hidden, device, temperature=1.0):
    with torch.no_grad():
        input_seq = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        output, hidden = model(input_seq, hidden)

        # Apply temperature to predictions
        output = output / temperature

        # Get the probabilities and sample from the distribution
        probs = torch.softmax(output, dim=-1).cpu().numpy()
        next_note = np.random.choice(len(probs[0]), p=probs[0])

        return next_note, hidden
