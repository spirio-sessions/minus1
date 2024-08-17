import json
import os


def convert_txt_to_json(directory):
    # Define the keys corresponding to the parameters
    keys = [
        "INPUT_SIZE", "hidden_size", "num_layers", "OUTPUT_SIZE", "learning_rate",
        "num_epochs", "batch_size", "seq_length", "stride", "databank", "data_cap",
        "train_loss", "val_loss"
    ]

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            txt_path = os.path.join(directory, filename)
            json_path = os.path.join(directory, os.path.splitext(filename)[0] + '.json')

            # Read the txt file
            with open(txt_path, 'r') as file:
                lines = file.read().splitlines()

            # Create a dictionary with keys and corresponding values from the txt file
            data_dict = {}
            for i, value in enumerate(lines):
                if i < len(keys):  # Make sure to only map available keys
                    data_dict[keys[i]] = value

            # Write the dictionary to a json file
            with open(json_path, 'w') as json_file:
                json.dump(data_dict, json_file, indent=4)

            print(f"Converted {filename} to {os.path.splitext(filename)[0]}.json")

# Example usage
convert_txt_to_json('G:\Schule\Studium\8. Semester\Bachelor-Minus1\minus1\pipeline_lstm_13keys\\04_finished_model\models')
