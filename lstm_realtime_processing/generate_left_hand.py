import torch


def generate_left_hand(input_array, model, hidden, device):
    """
    Generate left hand accompaniment using the trained LSTM model.
    """

    input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
    with torch.no_grad():
        output_tensor, hidden = model(input_tensor, hidden)
    output_array = output_tensor.squeeze(0).cpu().numpy()
    return output_array, hidden
