import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import shutil


def _find_all_projects(projects_dir: str):
    # Find all projects
    projects_path = Path(projects_dir)
    if not projects_path.exists():
        raise ValueError(f"The directory {projects_path} does not exist")

    # Find all project directories
    project_paths = [subdir for subdir in projects_path.iterdir() if subdir.is_dir()]
    print(f"Found {len(project_paths)} projects")

    return project_paths


def compute_val_loss_improvement(projects_dir: str):
    project_paths = _find_all_projects(projects_dir)
    improvements = []

    for project_path in project_paths:
        train_log = project_path / "training_log.txt"
        if not train_log.exists():
            raise ValueError(f"the file {train_log} does not exist in {project_path}")

        best_val_loss = float('inf')
        first_val_loss = None

        with open(train_log, 'r') as file:
            for line in file:
                parts = line.split(',')
                epoch = int(parts[0].split(':')[1].strip())
                train_loss = float(parts[1].split(':')[1].strip())
                val_loss = float(parts[2].split(':')[1].strip())

                if first_val_loss is None:
                    first_val_loss = val_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        if first_val_loss is not None:
            percentual_improvement = ((first_val_loss - best_val_loss) / first_val_loss) * 100
            improvements.append({
                "project_name": project_path.name,
                "first_val_loss": first_val_loss,
                "best_val_loss": best_val_loss,
                "improvement": percentual_improvement
            })

    # Sort improvements by percentual improvement in descending order
    sorted_improvements = sorted(improvements, key=lambda x: x["improvement"], reverse=True)

    # Print the improvements
    for improvement in sorted_improvements:
        print(f"Project {improvement['project_name']}: "
              f"First Val Loss = {improvement['first_val_loss']:.4f}, "
              f"Best Val Loss = {improvement['best_val_loss']:.4f}, "
              f"Improvement = {improvement['improvement']:.2f}%")

    return sorted_improvements
