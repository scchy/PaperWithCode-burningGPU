import numpy as np


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """_summary_

    Args:
        distance_df (_type_): data frame with three columns: [frame, to, distance]
        sensor_ids (_type_): list of sensor ids
        normalized_k (float, optional): entries that become lower than normalized_k after normalization are set to zero for sparsity. Defaults to 0.1.
    return
        adjacency matrix
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    sensor_id_to_ind = dict(zip(sensor_ids, range(len(sensor_ids))))
    # Fills cels in the matrix with distance
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
    
    # Caculates the standard deviation as theta
    distance = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distance.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshould, i.e, k, to zero for sparsity/
    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx
