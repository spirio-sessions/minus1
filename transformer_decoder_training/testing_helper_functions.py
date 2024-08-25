import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import shutil


def create_json_testing_template():
    # Create the template with the desired structure and empty values
    template = {
        "test_row_name": None,
        "max_context_length": None,
        "threshold": None,
        "initial_context_length": None,
        "inference_method": None
    }

    # Return the template object
    return template


def save_json_testing_configuration(config, project_path: Path, overwrite=False):
    # create project dir
    testing_dir = project_path / "tests"
    testing_dir.mkdir(exist_ok=overwrite)

    test_row_configuration_path = testing_dir / f"{config['test_row_name']}_configuration.json"
    with test_row_configuration_path.open('w') as json_file:
        json.dump(config, json_file, indent=4)

    # create test row dir
    test_row_dir = testing_dir / config["test_row_name"]
    test_row_dir.mkdir(exist_ok=True)

    return test_row_configuration_path


def copy_testing_templates_to_all_models(projects_dir: str, dir_with_configs: str, override=False):
    projects_path = Path(projects_dir)
    if not projects_path.exists():
        raise ValueError(f"The directory {projects_path} does not exist")

    configs_path = Path(dir_with_configs)
    if not configs_path.exists():
        raise ValueError(f"The directory {configs_path} does not exist")

    # Find all project directories
    project_paths = [subdir for subdir in projects_path.iterdir() if subdir.is_dir()]
    print(f"Found {len(project_paths)} projects")

    # Find all config files
    config_paths = list(configs_path.glob("*.json"))
    print(f"Found {len(config_paths)} config files")

    for project_path in project_paths:
        test_dir_path = project_path / "tests"

        if test_dir_path.exists():
            if override:
                print(f"Overriding existing test directory in {project_path}")
                shutil.rmtree(test_dir_path)
                test_dir_path.mkdir()
            else:
                print(f"{project_path} already has a test directory and override is False, skipping")
                continue
        else:
            test_dir_path.mkdir()

        for config_file in config_paths:
            destination_file = test_dir_path / config_file.name
            shutil.copyfile(config_file, destination_file)
            print(f"Copied {config_file} to {destination_file}")


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

        if len(snapshot_array_l) != len(snapshot_array_r):
            raise ValueError("Meloddy and harmony tracks have different lengths")

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

    # Initialize tqdm progress bar for testing configs
    progress_bar_testing_configs = tqdm(total=len(testing_configurations), desc="Testing configurations", leave=False)

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

        # Initialize tqdm progress bar for the songs
        progress_bar_sequences = tqdm(total=len(test_songs_dict.items()), desc="Songs",
                                      leave=False)

        # process each test song
        for songname, sequence in test_songs_dict.items():

            start_token = torch.tensor(config["training_data_params"]["sos_token"], dtype=torch.float32).to(device)
            pad_token = torch.tensor(config["training_data_params"]["pad_token"], dtype=torch.float32).to(device)

            sequence = torch.tensor(sequence, dtype=torch.float32).to(device)

            print("sequecne tensor:", sequence.shape)

            seq_with_start = torch.vstack((start_token, sequence))
            print("sequence with start token tensor:", seq_with_start.shape)

            context_seq, continuing_seq, original_seq = inference_and_visualize_1.prepare_sequence(seq_with_start,
                                                                                                   configuration[
                                                                                                       "initial_context_length"])
            print("context_seq tensor:", context_seq.shape)
            print("continuing seq", continuing_seq.shape)

            # save start time
            start_time = time.time()

            # select correct inference
            inference_method_name = configuration["inference_method"]
            if inference_method_name == "inference":
                tokens_with_truth, generated_logits = inference_5.inference(model,
                                                                            context_seq,
                                                                            continuing_seq,
                                                                            configuration["threshold"],
                                                                            pad_token,
                                                                            configuration["max_context_length"],
                                                                            device)
            elif inference_method_name == "inference_top_k_truth_notes":
                tokens_with_truth, generated_logits = inference_5.inference_top_k_truth_notes(model,
                                                                                              context_seq,
                                                                                              continuing_seq,
                                                                                              pad_token,
                                                                                              configuration[
                                                                                                  "max_context_length"],
                                                                                              device)
            else:
                raise ValueError(f"{inference_method_name} is no valid inference method")

            # Record the end time
            end_time = time.time()
            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds")

            generated_sequence = inference_and_visualize_1.combine_output_with_context(tokens_with_truth, context_seq)

            inference_and_visualize_1.inference_output_to_midi_one_octave(original_seq, context_seq,
                                                                          generated_sequence,
                                                                          config["training_data_params"][
                                                                              "snapshot_interval"],
                                                                          str(specific_configuration_dir),
                                                                          f"{songname}.mid")

            test_song_configuration = specific_configuration_dir / f"{songname}_configuration.json"

            specific_configuration = {
                "songname": songname,
                "max_context_length": configuration["max_context_length"],
                "threshold": configuration["threshold"],
                "inference_method_used": inference_method_name,
                "initial_context_length": configuration["initial_context_length"],
                "song_data": {
                    "sos_token": config["training_data_params"]["sos_token"],
                    "snapshot_interval": config["training_data_params"]["snapshot_interval"],
                    "song_length": sequence.shape[1],
                },
                "num_tokens_generated": continuing_seq.shape[1],
                "time_taken": elapsed_time,
                "generated_sequence": generated_sequence.tolist(),
                "generated_raw_logits": torch.cat(generated_logits, dim=0).tolist()
            }

            with test_song_configuration.open('w') as json_file:
                json.dump(specific_configuration, json_file, indent=4)

            progress_bar_sequences.update(1)

        progress_bar_sequences.close()
        progress_bar_testing_configs.update(1)

    progress_bar_testing_configs.close()


def testinference_for_all_models(projects_dir: str, test_data_dir: str, device):
    # Find all projects
    projects_path = Path(projects_dir)
    if not projects_path.exists():
        raise ValueError(f"The directory {projects_path} does not exist")

    test_data_path = Path(test_data_dir)
    if not test_data_path.exists():
        raise ValueError(f"The directory {test_data_path} does not exist")

    # Find all project directories
    project_paths = [subdir for subdir in projects_path.iterdir() if subdir.is_dir()]
    print(f"Found {len(project_paths)} projects")

    progress_bar_testing = tqdm(total=len(project_paths), desc="Projects",
                                leave=False)
    for project_path in project_paths:
        # load model
        try:
            model = load_transformer_model(project_path.name, projects_dir, device, load_best_checkpoint=True)
        except Exception as e:
            print(f"Could not load model from project {project_path}, skipping. Error: {e}")
            progress_bar_testing.update(1)
            continue

        # do testing
        try:
            testinference_for_model(model, project_path.name, projects_dir, device, test_data_dir)
        except Exception as e:
            print(f"Could not test model from project {project_path}, skipping. Error: {e}")
        progress_bar_testing.update(1)

    progress_bar_testing.close()
