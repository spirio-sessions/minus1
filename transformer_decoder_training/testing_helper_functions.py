import json
import torch
import numpy as np
from pathlib import Path


def _load_model(model_dict_path: str, parameters, device="cpu"):
    # Transformer without sigmoid output
    from transformer_decoder_training.models.transformer_decoder_2 import Transformer

    model = Transformer(num_emb=parameters["num_emb"], num_layers=parameters["num_layers"],
                        hidden_size=parameters["hidden_size"], num_heads=parameters["num_heads"]).to(device)
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


def _prepare_songs_for_testing(testing_dataset_dir: str, ):
    # load data
    dataset_as_snapshots = dataset_snapshot.process_dataset_multithreaded(dataset_dir, snapshot_intervall)
    # filter snapshots to 88 piano notes
    dataset_as_snapshots = dataset_snapshot.filter_piano_range(dataset_as_snapshots)
    # reduce to 12 keys
    dataset_as_snapshots = dataset_snapshot.compress_existing_dataset_to_12keys(dataset_as_snapshots)