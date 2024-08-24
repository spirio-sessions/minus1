import json
import torch
import numpy as np
from pathlib import Path


def create_json_template():
    # Create the template with the desired structure and empty values
    template = {
        "model_project_name": None,
        "model_params": {
            "model_topology": None,
            "num_emb": None,
            "hidden_size": None,
            "num_layers": None,
            "num_heads": None,
            "output_dim": None
        },
        "training_params": {
            "learning_rate": None,
            "num_epochs": None,
            "optimizer": "",
            "loss_fn": ""
        },
        "training_data_params": {
            "sos_token": [],
            "pad_token": [],
            "snapshot_interval": None,
            "batch_size": None,
            "sequence_length": None,
            "stride": None,
            "test_size": None
        }
    }

    # Return the template object
    return template


def print_json_parameters(data, indent=0):
    """
    Recursively prints JSON parameters.

    :param data: Dictionary loaded from JSON.
    :param indent: Current indentation level for nested parameters.
    """
    for key, value in data.items():
        print('  ' * indent + f"{key}: ", end="")
        if isinstance(value, dict):
            print()
            print_json_parameters(value, indent + 1)
        else:
            print(value)


def save_json_config(config, projects_path: Path, overwrite=False):
    # create project dir
    project_dir = projects_path / config["model_project_name"]
    project_dir.mkdir(exist_ok=overwrite)

    # Save the configuration file to the project directory
    config_path = project_dir / f"{config['model_project_name']}_config.json"
    with config_path.open('w') as json_file:
        json.dump(config, json_file, indent=4)

    return config_path


def prepare_training_data(config_file: str, dataset_dir: str):
    from transformer_decoder_training.dataprep_transformer.prepare_dataloader_complete import \
        prepare_dataset_as_dataloaders_no_test

    with open(config_file, 'r') as file:
        config = json.load(file)

    data_params = config["training_data_params"]

    train_loader, val_loader = prepare_dataset_as_dataloaders_no_test(dataset_dir,
                                                                      data_params["snapshot_interval"],
                                                                      data_params["batch_size"],
                                                                      data_params["sequence_length"],
                                                                      data_params["stride"],
                                                                      data_params["test_size"],
                                                                      np.array(data_params["sos_token"]))

    print(f"The train loader has {len(train_loader)} batches with a size of {data_params['batch_size']}")
    print(f"The model is trained on approximately {len(train_loader) * data_params["batch_size"]} sequences")

    return train_loader, val_loader


def _load_optimizer(model, config):
    import torch.optim as optim

    optimizer_name = config["training_params"]["optimizer"]
    learning_rate = config["training_params"]["learning_rate"]

    # Optional: Additional optimizer parameters
    optimizer_params = config["training_params"].get("optimizer_params", {})

    # Select the optimizer based on the name from the config
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, **optimizer_params)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, **optimizer_params)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_name}")

    return optimizer


def _load_loss_fn(config):
    import torch.nn as nn
    loss_fn_name = config["training_params"]["loss_fn"]

    # Select the loss function based on the name from the config
    if loss_fn_name == "CrossEntropyLoss":
        loss_fn = nn.CrossEntropyLoss()
    elif loss_fn_name == "MSELoss":
        loss_fn = nn.MSELoss()
    elif loss_fn_name == "BCEWithLogitsLoss":
        loss_fn = nn.BCEWithLogitsLoss()
    elif loss_fn_name == "MultiLabelSoftMarginLoss":
        loss_fn = nn.MultiLabelSoftMarginLoss()
    else:
        raise ValueError(f"Unsupported loss function type: {loss_fn_name}")

    return loss_fn


def _initialize_transformer(config, device):
    from transformer_decoder_training.models.transformer_decoder_2 import Transformer

    config = config["model_params"]

    model = Transformer(num_emb=config["num_emb"],
                        hidden_size=config["hidden_size"],
                        num_layers=config["num_layers"],
                        num_heads=config["num_heads"]).to(device)

    return model


def _train_model_epochs(model_dir, model, optimizer, loss_fn, pad_token, train_loader, val_loader, num_epochs, device):
    from timeit import default_timer as timer
    from transformer_decoder_training.training import training_1
    from data_visualization.Visualization import plot_losses

    # Create dir for model
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(exist_ok=True)

    # Create checkpoints dir
    checkpoints_dir_path = model_dir_path / "checkpoints"
    checkpoints_dir_path.mkdir(exist_ok=True)

    # Initialize lists to store loss values
    train_losses = []
    val_losses = []

    # Open a text file to save print outputs
    log_file_path = model_dir_path / "training_log.txt"

    # Delete the log file if it already exists
    if log_file_path.exists():
        log_file_path.unlink()

    for epoch in range(1, num_epochs + 1):
        start_time = timer()
        train_loss = training_1.train_loop(model, optimizer, loss_fn, train_loader, pad_token, device)
        end_time = timer()
        val_loss = training_1.validation_loop(model, loss_fn, val_loader, pad_token, device)

        # Log epoch information to console and file
        epoch_info = (f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                      f"Epoch time = {(end_time - start_time):.3f}s")
        print(epoch_info)

        # Open log file in append mode and write the log for this epoch
        with log_file_path.open("a") as log_file:
            log_file.write(epoch_info + "\n")

        # Store the loss values
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save checkpoint and plot for this epoch
        checkpoint_path = checkpoints_dir_path / f"checkpoint_epoch_{epoch}.pth"
        plot_path = checkpoints_dir_path / f"loss_plot_epoch_{epoch}.png"

        # Save model state
        torch.save(model.state_dict(), checkpoint_path)

        # Save loss plot
        plot_losses(train_losses, val_losses, str(plot_path))

    # Save final model and plot after all epochs
    final_model_path = model_dir_path / "final_model.pth"
    final_plot_path = model_dir_path / "final_loss_plot.png"

    # Save final model state
    torch.save(model.state_dict(), final_model_path)

    # Save final loss plot
    plot_losses(train_losses, val_losses, str(final_plot_path))


def train_model_from_config(config_file: str, dataset_dir, device):
    with open(config_file, 'r') as file:
        config = json.load(file)

    model_project_dir = Path(config_file).parent

    # load training data
    train_loader, val_loader = prepare_training_data(config_file, dataset_dir)

    #initialize training
    model = _initialize_transformer(config, device)
    optimizer = _load_optimizer(model, config)
    loss_fn = _load_loss_fn(config)

    # update json with model topology
    config["model_params"]["model_topology"] = str(model)
    save_json_config(config, model_project_dir.parent, overwrite=True)

    print("Start training Model with following parameters:")
    print("=============================")
    print_json_parameters(config)
    print("==============================")

    # train model
    pad_token = torch.tensor(config["training_data_params"]["pad_token"]).to(device)

    _train_model_epochs(str(model_project_dir), model, optimizer, loss_fn, pad_token, train_loader, val_loader,
                        config["training_params"]["num_epochs"], device)

    # delete variables to clear ram
    del train_loader, val_loader, model
