import numpy as np


def count_key_presses(sequence_tensor):
    """
    Count how often each key is pressed overall, considering when a key is held to the end of the sequence.

    Args:
        sequence_tensor (np.array): A 2D array where each row is a snapshot over time.

    Returns:
        int: The total count of key presses divided by sequence length
    """
    num_keys = sequence_tensor.shape[1]
    key_presses = 0
    previous_snapshot = np.zeros(num_keys)

    for snapshot in sequence_tensor:
        # Count key presses (1 -> 0 transitions)
        key_presses += np.sum((previous_snapshot == 1.0) & (snapshot == 0.0))
        previous_snapshot = snapshot

    # Handle the case where keys are held to the end of the sequence
    key_presses += np.sum(previous_snapshot == 1.0)

    return key_presses


import numpy as np
import itertools

# Define common harmonic intervals and chords (in semitones)
harmonic_intervals = [
    {0, 7},  # Perfect fifth
    {0, 4},  # Major third
    {0, 3},  # Minor third
    {0, 5},  # Perfect fourth
]

chord_intervals = [
    {0, 4, 7},  # Major triad
    {0, 3, 7},  # Minor triad
    {0, 3, 6},  # Diminished triad
    {0, 4, 8},  # Augmented triad
]

# Define dissonant intervals
dissonant_intervals = [
    {0, 1},  # Minor second
    {0, 6},  # Tritone
    {0, 10},  # Minor seventh
]


def detect_harmony_and_disharmony_score(snapshot):
    """
    Detect if a snapshot contains harmonic intervals, chords, or dissonant intervals,
    and return a score based on the number and type of harmonies and disharmonies detected.

    Args:
        snapshot (list or np.array): A list or array representing a snapshot of notes (1.0 if active, 0.0 if not).

    Returns:
        dict: A dictionary with scores for harmony and disharmony.
    """
    active_notes = np.where(np.array(snapshot) == 1.0)[0]
    if len(active_notes) < 2:
        return {'harmony_score': 0, 'disharmony_score': 0}  # Need at least 2 notes to score

    harmony_score = 0
    disharmony_score = 0
    disharmony_detected = False

    # Check all possible pairs and combinations of active notes for harmonic and dissonant intervals
    for combination_size in range(2, len(active_notes) + 1):
        for combination in itertools.combinations(active_notes, combination_size):
            intervals = set((np.array(combination) - combination[0]) % 12)

            # Check for dissonant intervals first
            for dissonant in dissonant_intervals:
                if dissonant.issubset(intervals):
                    disharmony_score += 1  # Increase disharmony score for each detected dissonant interval
                    disharmony_detected = True

            # Check for harmonic intervals only if no disharmony detected yet
            if not disharmony_detected:
                for harmonic in harmonic_intervals:
                    if harmonic.issubset(intervals):
                        harmony_score += 1  # Increase harmony score for each detected harmonic interval

                # Check for chords
                for chord in chord_intervals:
                    if chord.issubset(intervals):
                        harmony_score += 2  # Increase harmony score more for detected chords

    return {'harmony_score': harmony_score, 'disharmony_score': disharmony_score}


def compute_quality_scores(sequence_tensor):
    """
    Compute individual quality scores for harmonic richness, dissonance, note density, and pattern variation.

    Args:
        sequence_tensor (np.array): A 2D array where each row is a snapshot over time.

    Returns:
        dict: A dictionary containing the individual quality scores.
    """

    notes_per_snapshot = count_key_presses(sequence_tensor)
    harmony_score = 0
    disharmony_score = 0

    for snapshot in sequence_tensor:
        temp_scores = detect_harmony_and_disharmony_score(snapshot)
        harmony_score += temp_scores["harmony_score"]
        disharmony_score += temp_scores["disharmony_score"]

    notes_per_snapshot = notes_per_snapshot / sequence_tensor.shape[0]
    harmony_score = harmony_score / sequence_tensor.shape[0]
    disharmony_score = disharmony_score / sequence_tensor.shape[0]

    return {'harmony_score': harmony_score, 'disharmony_score': disharmony_score, "notes_per_snapshot": notes_per_snapshot}