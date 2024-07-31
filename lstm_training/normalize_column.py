import numpy as np


def normalize_column(column, start_index=1024):
    subset = column[start_index:]
    sorted_values = np.sort(subset.unique())
    second_lowest = sorted_values[1]
    highest = sorted_values[-1]

    def normalize(value):
        if value == 0 or value == 1:
            return value
        if highest != second_lowest:
            return (value - second_lowest) / (highest - second_lowest)
        else:
            return 0
    return column.apply(normalize)
