import torch


def train_loop(model, opt, loss_fn, dataloader, pad_token, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Move data to GPU
        src_sequence = batch.to(device)

        # create input and expected sequence -> move expected sequence one to the right
        input_sequences = src_sequence[:, :-1]
        expected_sequence = src_sequence[:, 1:]

        # Generate predictions
        pred = model(input_sequences, pad_token)

        # print("Prediction shape:", pred.shape)
        # print(pred)
        # print("expected harmony_shape:", expected_harmony.shape)
        # print(expected_harmony)

        # Calculate loss with masked cross-entropy
        # ich glaube 0 steht in vorlage für padding token index -> habe ich hier anders
        # mask = (expected_harmony != pad_token).float() Maske verwenden, um Padding positions im output zu canceln
        # masked_pred = pred * mask
        loss = loss_fn(pred, expected_sequence)

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader, pad_token, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move data to GPU
            src_sequence = batch.to(device)

            # Create input and expected sequences
            input_sequences = src_sequence[:, :-1, :]
            expected_sequence = src_sequence[:, 1:, :]

            # Generate predictions
            pred = model(input_sequences, pad_token)

            # Calculate loss without flattening
            loss = loss_fn(pred, expected_sequence)

            total_loss += loss.detach().item()

    return total_loss / len(dataloader)