import torch

def inference(model, sequence, num_of_tokens_to_generate, threshold, pad_token, device):
    generated_tokens = []
    # clone seq so we dont change the original input
    input_seq = sequence.clone()
    with torch.no_grad():
        for i in range(num_of_tokens_to_generate):
            input_seq.to(device)


            # model prediction for new token
            data_pred = model(input_seq, pad_token)


            #print("output shape of model for one prediction", data_pred.shape)

            if data_pred.shape[1] > (input_seq.shape[1]):
                print("Model might have generated more than one token")

            # get last token from otuput (should be the one new token)
            next_token = data_pred[:, -1, :]
            # print("next token shape:", next_token.shape)
            # Change probability values to binary with threshold
            next_token = (next_token >= threshold).float()

            # add token to list
            generated_tokens.append(next_token)
            # Append the new token to the sequence
            input_seq = torch.cat((input_seq, next_token.unsqueeze(1)), dim=1)

    return generated_tokens