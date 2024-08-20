import json
import torch
import numpy as np
from pathlib import Path


def create_json_testing_template():
    # Create the template with the desired structure and empty values
    template = {
        "test_row_name": None,
        "max_context_length": None,
        "threshold": None,
        "initial_context_length": None
    }

    # Return the template object
    return template


def save_json_testing_configuration(config, project_path: Path):
    # create project dir
    testing_dir = project_path / "tests"
    testing_dir.mkdir(exist_ok=True)

    test_row_configuration_path = testing_dir / f"{config['test_row_name']}_configuration.json"
    with test_row_configuration_path.open('w') as json_file:
        json.dump(config, json_file, indent=4)

    # create test row dir
    test_row_dir = testing_dir / config["test_row_name"]
    test_row_dir.mkdir(exist_ok=True)

    return test_row_configuration_path


def _load_model(model_dict_path: str, parameters, device="cpu"):
    # Transformer without sigmoid output
    from transformer_decoder_training.models.transformer_decoder_2 import Transformer

    model_params = parameters["model_params"]

    model = Transformer(num_emb=model_params["num_emb"], num_layers=model_params["num_layers"],
                        hidden_size=model_params["hidden_size"], num_heads=model_params["num_heads"]).to(device)
    model.load_state_dict(torch.load(model_dict_path))

    model.eval()

    return model


def _find_best_epoch(file_path: str):
    best_epoch = None
    best_val_loss = float('inf')
    best_train_loss = float('inf')

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(',')
            epoch = int(parts[0].split(':')[1].strip())
            train_loss = float(parts[1].split(':')[1].strip())
            val_loss = float(parts[2].split(':')[1].strip())

            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
                best_train_loss = train_loss
            elif val_loss == best_val_loss:
                if train_loss < best_train_loss:
                    best_epoch = epoch
                    best_train_loss = train_loss

    return best_epoch


def load_transformer_model(model_project_name: str, projects_dir: str, device: str, load_best_checkpoint=False,
                           specific_epoch=-1):
    project_path = Path(projects_dir) / model_project_name
    if not project_path.exists():
        raise ValueError(f"the project {model_project_name} does not exist in {projects_dir}")

    config_path = list(project_path.glob("*_config.json"))
    if len(config_path) == 0:
        raise FileNotFoundError("No JSON files found in the directory.")
    elif len(config_path) > 1:
        raise ValueError("More than one JSON file found in the directory.")
    config_path = config_path[0]

    with open(config_path, 'r') as file:
        config = json.load(file)

    # Load the best checkpoint
    if load_best_checkpoint:
        train_log = Path(project_path) / "training_log.txt"
        if not train_log.exists():
            raise ValueError(f"the file {model_project_name} does not exist in {project_path}")

        best_epoch = _find_best_epoch(str(train_log))
        print(f"Best epoch is epoch number {best_epoch}")

        checkpoints_path = project_path / "checkpoints"
        if not checkpoints_path.exists():
            raise ValueError(f"{checkpoints_path} does not exist")

        best_checkpoint_path = list(checkpoints_path.glob(f"*epoch_{best_epoch}.pth"))

        if len(best_checkpoint_path) != 1:
            raise ValueError(f"{checkpoints_path} does not exactly math a file containing {best_epoch}")

        best_checkpoint_path = best_checkpoint_path[0]

        model = _load_model(str(best_checkpoint_path), config, device)
        return model

    # Load a specific checkpoint
    if specific_epoch > 0:
        checkpoints_path = project_path / "checkpoints"
        if not checkpoints_path.exists():
            raise ValueError(f"{checkpoints_path} does not exist")

        specific_epoch_checkpoint_path = list(checkpoints_path.glob(f"*epoch_{specific_epoch}.pth"))
        if len(specific_epoch_checkpoint_path) != 1:
            raise ValueError(f"{checkpoints_path} does not exactly math a file containing {specific_epoch}")
        specific_epoch_checkpoint_path = specific_epoch_checkpoint_path[0]

        model = _load_model(str(specific_epoch_checkpoint_path), config, device)
        return model

    # Load the final checkpoint
    model_dict_path = project_path / "final_model.pth"
    if not project_path.exists():
        raise ValueError(f"{model_dict_path} does not exist in {project_path}")

    model = _load_model(str(model_dict_path), config, device)
    return model


def _prepare_songs_for_testing(testing_dataset_dir: str, model_config):
    """
        Prepares songs for testing by processing MIDI files and creating snapshots for both hands.

        This function processes MIDI files found in the specified testing dataset directory,
        generates snapshots based on the given interval, and combines the left and right hand
        snapshots for each song. The resulting data is returned in a dictionary with song names
        as keys and the combined snapshot arrays as values.

        Args:
            testing_dataset_dir (str): The directory containing the MIDI files to be tested.
            model_config (dict): A dictionary containing the model configuration parameters.
                                 This includes the "snapshot_intervall" under
                                 "training_data_params" which defines the interval for
                                 generating snapshots.

        Returns:
            dict: A dictionary where each key is a song name, and the value is a combined
                  snapshot array of the left and right hand MIDI data.

        Notes:
            - The function assumes that the `dataset_snapshot.find_midi_files` function returns
              a dictionary where each key is a song name and the value is a dictionary with
              'leftH' and 'rightH' keys corresponding to the left and right hand MIDI file paths.
            - The snapshots for each hand are generated using the `dataset_snapshot.__process_single_midi`
              function, which is expected to return both the file path and the snapshot array.
              The snapshot arrays for both hands are then concatenated along the second axis.

    """
    from data_preperation import dataset_snapshot

    snapshot_intervall = model_config["training_data_params"]["snapshot_interval"]

    midi_files = dataset_snapshot.find_midi_files(testing_dataset_dir)

    test_songs_dict = {}

    for song, hands_dict in midi_files.items():
        # song snapshots should have left hand first, then right hand
        left_h = hands_dict["leftH"]
        right_h = hands_dict["rightH"]

        midi_file_path_l, snapshot_array_l = dataset_snapshot.__process_single_midi(left_h, snapshot_intervall)
        midi_file_path_r, snapshot_array_r = dataset_snapshot.__process_single_midi(right_h, snapshot_intervall)

        snapshot_array_l = dataset_snapshot.compress_track(snapshot_array_l)
        snapshot_array_r = dataset_snapshot.compress_track(snapshot_array_r)

        combined = np.concatenate((snapshot_array_l, snapshot_array_r), axis=1)

        test_songs_dict[song] = combined

    return test_songs_dict


def testinference_for_model(model, model_project_name: str, projects_dir: str, device: str, testing_dataset_dir: str):
    from transformer_decoder_training.transformer_inference_eval import inference_and_visualize_1
    from transformer_decoder_training.inference import inference_5

    project_path = Path(projects_dir) / model_project_name
    if not project_path.exists():
        raise ValueError(f"the project {model_project_name} does not exist in {projects_dir}")

    config_path = list(project_path.glob("*_config.json"))
    if len(config_path) == 0:
        raise FileNotFoundError("No JSON files found in the directory.")
    elif len(config_path) > 1:
        raise ValueError("More than one JSON file found in the directory.")
    config_path = config_path[0]

    with open(config_path, 'r') as file:
        config = json.load(file)

    test_songs_dict = _prepare_songs_for_testing(testing_dataset_dir, config)

    # test each song and save testing configuration

    # find all testing configurations
    testing_path = project_path / "tests"
    if not project_path.exists():
        raise ValueError(f"the testing dir {testing_path} does not exist in {project_path}")

    testing_configurations = list(testing_path.glob("*_configuration.json"))

    for configuration_path in testing_configurations:
        # open each configuration file
        with open(configuration_path, 'r') as file:
            configuration = json.load(file)
        # enter specific configuration dir
        specific_configuration_dir = testing_path / configuration["test_row_name"]
        specific_configuration_dir.mkdir(exist_ok=True)

        # set testing parameters if not specified
        if configuration["max_context_length"] is None:
            configuration["max_context_length"] = config["training_data_params"]["sequence_length"] + 1

        if configuration["initial_context_length"] is None:
            configuration["initial_context_length"] = config["training_data_params"]["sequence_length"] + 1

        # process each test song
        for songname, sequence in test_songs_dict.items():

            start_token = torch.tensor(config["training_data_params"]["sos_token"], dtype=torch.float32).to(device)
            pad_token = torch.tensor(config["training_data_params"]["pad_token"], dtype=torch.float32).to(device)

            sequence = torch.tensor(sequence, dtype=torch.float32).to(device)

            print("sequecne tensor:", sequence.shape)

            seq_with_start = torch.vstack((start_token, sequence))

            context_seq, continuing_seq, original_seq = inference_and_visualize_1.prepare_sequence(seq_with_start,
                                                                                                   configuration["initial_context_length"])
            tokens_with_truth, generated_logits = inference_5.inference(model,
                                                                        context_seq,
                                                                        continuing_seq,
                                                                        configuration["threshold"],
                                                                        pad_token,
                                                                        configuration["max_context_length"],
                                                                        device)

            generated_sequence = inference_and_visualize_1.combine_output_with_context(tokens_with_truth, context_seq)

            inference_and_visualize_1.inference_output_to_midi_one_octave(original_seq, context_seq,
                                                                          generated_sequence,
                                                                          config["training_data_params"]["snapshot_interval"],
                                                                          str(specific_configuration_dir),
                                                                          f"{songname}.mid")

            test_song_configuration = specific_configuration_dir / f"{songname}_configuration.json"

            specific_configuration = {
                "songname": songname,
                "max_context_length": configuration["max_context_length"],
                "threshold": configuration["threshold"],
                "initial_context_length": configuration["initial_context_length"],
                "song_data": {
                    "sos_token": config["training_data_params"]["sos_token"],
                    "snapshot_interval": config["training_data_params"]["snapshot_interval"],
                    "song_length": sequence.shape[1],
                }
            }

            with test_song_configuration.open('w') as json_file:
                json.dump(specific_configuration, json_file, indent=4)
