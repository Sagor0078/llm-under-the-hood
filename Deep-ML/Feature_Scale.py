import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    
    data = np.array(data, dtype=float)

    # Z-score normalization
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std

    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)

    standardized_data = np.round(standardized_data, 4)
    normalized_data = np.round(normalized_data, 4)

    return standardized_data, normalized_data


# Problem Link : https://www.deep-ml.com/problems/16