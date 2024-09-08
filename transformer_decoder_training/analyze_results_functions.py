import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import shutil
import transformer_decoder_training.music_score_computation as music_score


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
        best_epoch = None

        with open(train_log, 'r') as file:
            for line in file:
                parts = line.split(',')
                epoch = int(parts[0].split(':')[1].strip())
                val_loss = float(parts[2].split(':')[1].strip())

                if first_val_loss is None:
                    first_val_loss = val_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch

        if first_val_loss is not None:
            percentual_improvement = ((first_val_loss - best_val_loss) / first_val_loss) * 100
            improvements.append({
                "project_name": project_path.name,
                "first_val_loss": first_val_loss,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "improvement": percentual_improvement
            })

    # Sort improvements by percentual improvement in descending order
    sorted_improvements = sorted(improvements, key=lambda x: x["improvement"], reverse=True)

    # Print the improvements
    for improvement in sorted_improvements:
        print(f"Project {improvement['project_name']}: "
              f"First Val Loss = {improvement['first_val_loss']:.4f}, "
              f"Best Val Loss = {improvement['best_val_loss']:.4f} "
              f"(Epoch {improvement['best_epoch']}), "
              f"Improvement = {improvement['improvement']:.2f}%")

    return sorted_improvements


def _load_generated_snapshots(json_file: Path):
    """Loads the generated snapshots and logits from a JSON file and returns them as tensors."""
    if not json_file.exists():
        raise ValueError(f"File: {json_file} does not exist")

    with open(json_file, 'r') as file:
        config = json.load(file)

    raw_logits = config["generated_raw_logits"]
    generated_sequence = config["generated_sequence"]

    raw_logits = torch.tensor(raw_logits, dtype=torch.float32)
    generated_sequence = torch.tensor(generated_sequence).squeeze(dim=0)

    # Extract relevant information from the JSON
    initial_context_length = config.get("initial_context_length", 0)

    # Only analyze the generated tokens (exclude initial context)
    raw_logits = raw_logits[initial_context_length:]
    generated_sequence = generated_sequence[initial_context_length:]

    return generated_sequence, raw_logits, config


def analyze_snapshots_for_project(project_dir: str, score_weights=None):
    project_dir = Path(project_dir)

    test_dir = project_dir / "tests"
    if not test_dir.exists():
        raise ValueError("Test dir does not exist")

    test_rows = [p for p in test_dir.iterdir() if p.is_dir()]

    music_score_evaluation = {
        "score_weights": score_weights,
        "test_rows": []
    }

    for test_row in test_rows:
        print(f"Analyzing {test_row.name}...")

        json_files = test_row.glob("*.json")
        song_scores = []

        for json_file in json_files:
            print(f"  Processing file: {json_file.name}")
            generated_sequence, raw_logits, config = _load_generated_snapshots(json_file)

            # Extract only the left-hand notes (first half of the note dimensions)
            left_hand_sequence = generated_sequence[:, :generated_sequence.size(1) // 2].numpy()

            # Apply sigmoid to raw logits to obtain probabilities (for left-hand logits only)
            left_hand_logits = raw_logits[:, :raw_logits.size(1) // 2]
            probabilities = torch.sigmoid(left_hand_logits).numpy()

            # Compute the individual scores for the sequence using the provided functions
            scores = music_score.compute_quality_scores(left_hand_sequence)

            # Combine harmony and disharmony scores
            combined_harmony_score = scores['harmony_score'] - scores['disharmony_score']

            song_scores.append({
                "song_name": json_file.stem,
                "single_harmony_score": scores['harmony_score'],
                "single_disharmony_score": scores['disharmony_score'],
                "combined_harmony_score": combined_harmony_score,
                "notes_per_snapshot_score": scores['notes_per_snapshot'] * 100
            })

        # Calculate the average scores for the test row
        harmony_average = sum(song['combined_harmony_score'] for song in song_scores) / len(song_scores) if song_scores else 0
        notes_per_snapshot_average = sum(song['notes_per_snapshot_score'] for song in song_scores) / len(song_scores) if song_scores else 0
        harmony_single_average = sum(song['single_harmony_score'] for song in song_scores) / len(
            song_scores) if song_scores else 0
        disharmony_single_average = sum(song['single_disharmony_score'] for song in song_scores) / len(
            song_scores) if song_scores else 0

        # Append the result to the music_score_evaluation structure
        music_score_evaluation["test_rows"].append({
            "name": test_row.name,
            "songs": song_scores,
            "single_harmony_average": harmony_single_average,
            "single_disharmony_average": disharmony_single_average,
            "harmony_average": harmony_average,
            "notes_per_snapshot_average": notes_per_snapshot_average
        })

    # Save the evaluation results to a JSON file
    eval_json_path = project_dir / "music_evaluation.json"
    with eval_json_path.open('w') as json_file:
        json.dump(music_score_evaluation, json_file, indent=4)

    print(f"Evaluation results saved to {eval_json_path}")


def analyze_all_projects(projects_dir: str):
    project_paths = _find_all_projects(projects_dir)

    for project_path in project_paths:
        print(f"analyzing project {project_path.name}")
        analyze_snapshots_for_project(str(project_path))

