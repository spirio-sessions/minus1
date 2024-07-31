def normalize_each_column_itself(column, start_index=1024):
    subset = column[(column != 0 & (column != 1))]
    min_val = subset.min()
    max_val = subset.max()

    def normalize(value):
        if value == 0 or value == 1:
            return value
        if max_val != min_val:
            return (value - min_val) / (max_val - min_val)
        else:
            return 0
    return column.apply(normalize)

