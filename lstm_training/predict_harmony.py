import numpy as np
import torch


def predict_harmony(model, melody):
    # Check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    harmonies = []
    with torch.no_grad():
        for i in range(melody.shape[0]):
            single_melody = torch.tensor(melody[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            harmony = model(single_melody)
            harmonies.append(harmony.squeeze(0).cpu().numpy())
    return np.array(harmonies)
