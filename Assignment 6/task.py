import sys
import time
from sklearn.cluster import KMeans
import numpy as np

def read_dataset(file_path):
    """
    Read the dataset from the given file path.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        The numpy array of data points
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        # Parse each line and convert to float values
        values = line.strip('\n').split(',')
        data.append(values)
    
    # Convert to numpy array and ensure correct data type
    return np.array(data).astype(np.float64)

def compute_mahalanobis_distance(point, centroid, std_dev):
    """
    Compute Mahalanobis distance between a point and a cluster
    
    Args:
        point: The data point
        centroid: The centroid of the cluster
        std_dev: The standard deviation of the cluster
        
    Returns:
        The Mahalanobis distance
    """
    # Handle zero standard deviations to avoid division by zero
    valid_indices = std_dev != 0
    
    if not np.any(valid_indices):
        return float('inf')
    
    # Calculate squared normalized deviations only for non-zero std_dev dimensions
    squared_deviations = np.zeros_like(point)
    squared_deviations[valid_indices] = np.square(
        (point[valid_indices] - centroid[valid_indices]) / std_dev[valid_indices]
    )
    
    # Mahalanobis distance is the square root of the sum of squared normalized deviations
    return np.sqrt(np.sum(squared_deviations))

def update_cluster_stats(n, SUM, SUMSQ, new_point):
    """
    Update cluster statistics (N, SUM, SUMSQ) when adding a new point
    
    Args:
        n: Current number of points in the cluster
        SUM: Current sum of coordinates
        SUMSQ: Current sum of squared coordinates
        new_point: New point to add to the cluster
        
    Returns:
        Updated n, SUM, SUMSQ, centroid, and standard deviation
    """
    n_new = n + 1
    SUM_new = np.add(SUM, new_point)
    SUMSQ_new = np.add(SUMSQ, np.square(new_point))
    
    # Update centroid and standard deviation
    centroid_new = SUM_new / n_new
    std_dev_new = np.sqrt(np.subtract(SUMSQ_new / n_new, np.square(centroid_new)))
    
    return n_new, SUM_new, SUMSQ_new, centroid_new, std_dev_new

def merge_clusters(n1, SUM1, SUMSQ1, n2, SUM2, SUMSQ2):
    """
    Merge two clusters and update statistics
    
    Args:
        n1, SUM1, SUMSQ1: Statistics of first cluster
        n2, SUM2, SUMSQ2: Statistics of second cluster
        
    Returns:
        Merged statistics and updated centroid and standard deviation
    """
    n_merged = n1 + n2
    SUM_merged = np.add(SUM1, SUM2)
    SUMSQ_merged = np.add(SUMSQ1, SUMSQ2)
    
    # Calculate new centroid and standard deviation
    centroid_merged = SUM_merged / n_merged
    std_dev_merged = np.sqrt(np.subtract(SUMSQ_merged / n_merged, np.square(centroid_merged)))
    
    return n_merged, SUM_merged, SUMSQ_merged, centroid_merged, std_dev_merged

def cluster_with_bfr(input_path, n_cluster, output_path):
    """
    Implement the BFR clustering algorithm
    
    Args:
        input_path: Path to the input dataset
        n_cluster: Number of clusters to form
        output_path: Path to write the output results
    """
    start_time = time.time()
    
    # Initialize data structures for Discard Set (DS), Compression Set (CS), and Retained Set (RS)
    rs = set()                # Outliers (points not yet assigned to any cluster)
    ds_dict = {}              # Statistics for DS clusters
    ds_centroid_dict = {}     # Centroids for DS clusters
    ds_std_dev_dict = {}      # Standard deviations for DS clusters
    ds_point_dict = {}        # Points belonging to each DS cluster
    cs_dict = {}              # Statistics for CS clusters
    cs_centroid_dict = {}     # Centroids for CS clusters
    cs_std_dev_dict = {}      # Standard deviations for CS clusters
    cs_point_dict = {}        # Points belonging to each CS cluster
    
    # Distance threshold multiplier
    distance_threshold = 2
    
    # Read the dataset
    full_data = read_dataset(input_path)
    
    # Randomly shuffle and split the data into 5 chunks (20% each)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(full_data)
    data_chunks = np.array_split(full_data, 5)
    
    # STEP 1: Load the first 20% of data
    data_chunk1 = data_chunks[0]
    
    # STEP 2: Run K-means with large K (5 * n_cluster)
    large_k = 5 * n_cluster
    kmeans_large = KMeans(n_clusters=large_k, random_state=42).fit(data_chunk1[:, 2:])
    
    # STEP 3: Move singletons to RS (outliers)
    cluster_counts = {}
    for idx, cluster_id in enumerate(kmeans_large.labels_):
        if cluster_id not in cluster_counts:
            cluster_counts[cluster_id] = []
        cluster_counts[cluster_id].append(idx)
    
    # Identify singleton clusters
    rs_indices = set()  # Local indices for RS points
    for cluster_id, point_indices in cluster_counts.items():
        if len(point_indices) == 1:
            rs_indices.add(point_indices[0])
    
    # Keep non-singleton points for DS formation
    non_singleton_indices = np.array([i for i in range(len(data_chunk1)) if i not in rs_indices])
    ds_candidate_points = data_chunk1[non_singleton_indices]
    
    # STEP 4: Run K-means with K = n_cluster on non-singleton points
    if len(ds_candidate_points) >= n_cluster:
        kmeans_final = KMeans(n_clusters=n_cluster, random_state=42).fit(ds_candidate_points[:, 2:])
        
        # STEP 5: Generate DS clusters from the K-means result
        ds_clusters = {}
        for idx, cluster_id in enumerate(kmeans_final.labels_):
            if cluster_id not in ds_clusters:
                ds_clusters[cluster_id] = []
            ds_clusters[cluster_id].append(idx)
        
        # Calculate statistics for each DS cluster
        for cluster_id, point_indices in ds_clusters.items():
            # Extract feature values (excluding index and cluster ID columns)
            features = ds_candidate_points[point_indices, 2:]
            n = len(point_indices)
            SUM = np.sum(features, axis=0)
            SUMSQ = np.sum(np.square(features), axis=0)
            
            # Store statistics
            ds_dict[cluster_id] = [n, SUM, SUMSQ]
            
            # Calculate centroid and standard deviation
            centroid = SUM / n
            std_dev = np.sqrt(np.subtract(SUMSQ / n, np.square(centroid)))
            
            ds_centroid_dict[cluster_id] = centroid
            ds_std_dev_dict[cluster_id] = std_dev
            
            # Store point IDs (from first column of data)
            point_ids = ds_candidate_points[point_indices, 0].astype(int).tolist()
            ds_point_dict[cluster_id] = point_ids
    
    # STEP 6: Try to form CS clusters from RS if there are enough points
    rs_points = data_chunk1[list(rs_indices)]
    # Store original point IDs in RS set
    for idx in rs_indices:
        rs.add(int(data_chunk1[idx, 0]))
    
    if len(rs_indices) >= large_k:
        # Run K-means on RS points
        kmeans_rs = KMeans(n_clusters=large_k, random_state=42).fit(rs_points[:, 2:])
        
        # Identify new clusters and singletons
        rs_clusters = {}
        for idx, cluster_id in enumerate(kmeans_rs.labels_):
            if cluster_id not in rs_clusters:
                rs_clusters[cluster_id] = []
            rs_clusters[cluster_id].append(idx)
        
        # Process clusters from RS
        new_rs = set()  # New RS set with original point IDs
        for cluster_id, point_indices in rs_clusters.items():
            if len(point_indices) == 1:
                # Keep singleton in RS (with original point ID)
                idx = point_indices[0]
                original_point_id = int(rs_points[idx, 0])
                new_rs.add(original_point_id)
                continue
                
            # For non-singleton clusters, create CS clusters
            features = rs_points[point_indices, 2:]
            n = len(point_indices)
            SUM = np.sum(features, axis=0)
            SUMSQ = np.sum(np.square(features), axis=0)
            
            # Store statistics for CS
            cs_dict[cluster_id] = [n, SUM, SUMSQ]
            
            # Calculate centroid and standard deviation
            centroid = SUM / n
            std_dev = np.sqrt(np.subtract(SUMSQ / n, np.square(centroid)))
            
            cs_centroid_dict[cluster_id] = centroid
            cs_std_dev_dict[cluster_id] = std_dev
            
            # Store point IDs
            point_ids = rs_points[point_indices, 0].astype(int).tolist()
            cs_point_dict[cluster_id] = point_ids
            
            # Remove these points from RS
            for point_id in point_ids:
                if point_id in rs:
                    rs.remove(point_id)
        
        # Update RS with new singleton IDs
        rs = new_rs
    
    # Write first round results to output file
    with open(output_path, "w") as f:
        f.write('The intermediate results:\n')
        
        # Count points in each set
        num_ds_points = sum(stats[0] for stats in ds_dict.values())
        num_cs_clusters = len(cs_dict)
        num_cs_points = sum(stats[0] for stats in cs_dict.values())
        num_rs_points = len(rs)
        
        result_line = f'Round 1: {num_ds_points},{num_cs_clusters},{num_cs_points},{num_rs_points}\n'
        f.write(result_line)
    
    # Process remaining chunks (rounds 2-5)
    # Calculate dimension of the data (for Mahalanobis distance threshold)
    data_dimension = data_chunk1.shape[1] - 2  # Exclude index and cluster ID columns
    mahala_threshold = distance_threshold * np.sqrt(data_dimension)
    
    for round_num in range(2, 6):
        # STEP 7: Load the next chunk of data
        current_chunk = data_chunks[round_num - 1]
        current_rs_indices = set()  # Store indices within current chunk
        
        # STEP 8-10: Assign each point to DS, CS, or RS
        for idx, point in enumerate(current_chunk):
            point_features = point[2:]  # Extract features
            
            # STEP 8: Try to assign to nearest DS cluster
            min_distance = float('inf')
            nearest_ds = None
            
            for cluster_id in ds_dict:
                distance = compute_mahalanobis_distance(
                    point_features, 
                    ds_centroid_dict[cluster_id], 
                    ds_std_dev_dict[cluster_id]
                )
                if distance < min_distance:
                    min_distance = distance
                    nearest_ds = cluster_id
            
            # If close enough to a DS cluster, assign to it
            if min_distance < mahala_threshold and nearest_ds is not None:
                # Update DS statistics
                n, SUM, SUMSQ = ds_dict[nearest_ds]
                n_new, SUM_new, SUMSQ_new, centroid_new, std_dev_new = update_cluster_stats(
                    n, SUM, SUMSQ, point_features
                )
                
                ds_dict[nearest_ds] = [n_new, SUM_new, SUMSQ_new]
                ds_centroid_dict[nearest_ds] = centroid_new
                ds_std_dev_dict[nearest_ds] = std_dev_new
                ds_point_dict[nearest_ds].append(int(point[0]))
                continue
            
            # STEP 9: If not assigned to DS, try to assign to nearest CS cluster
            min_distance = float('inf')
            nearest_cs = None
            
            for cluster_id in cs_dict:
                distance = compute_mahalanobis_distance(
                    point_features, 
                    cs_centroid_dict[cluster_id], 
                    cs_std_dev_dict[cluster_id]
                )
                if distance < min_distance:
                    min_distance = distance
                    nearest_cs = cluster_id
            
            # If close enough to a CS cluster, assign to it
            if min_distance < mahala_threshold and nearest_cs is not None:
                # Update CS statistics
                n, SUM, SUMSQ = cs_dict[nearest_cs]
                n_new, SUM_new, SUMSQ_new, centroid_new, std_dev_new = update_cluster_stats(
                    n, SUM, SUMSQ, point_features
                )
                
                cs_dict[nearest_cs] = [n_new, SUM_new, SUMSQ_new]
                cs_centroid_dict[nearest_cs] = centroid_new
                cs_std_dev_dict[nearest_cs] = std_dev_new
                cs_point_dict[nearest_cs].append(int(point[0]))
                continue
            
            # STEP 10: If not assigned to DS or CS, add to RS
            current_rs_indices.add(idx)
            rs.add(int(point[0]))  # Add point ID directly to the RS set
        
        # STEP 11: Process the RS points in current chunk if available
        if current_rs_indices:
            rs_points = current_chunk[list(current_rs_indices)]
            
            # If enough RS points, try to form new CS clusters
            if len(current_rs_indices) >= large_k:
                kmeans_rs = KMeans(n_clusters=large_k, random_state=42).fit(rs_points[:, 2:])
                
                rs_clusters = {}
                for idx, cluster_id in enumerate(kmeans_rs.labels_):
                    if cluster_id not in rs_clusters:
                        rs_clusters[cluster_id] = []
                    rs_clusters[cluster_id].append(idx)
                
                # Process clusters from current RS points
                for cluster_id, point_indices in rs_clusters.items():
                    if len(point_indices) == 1:
                        # Skip singletons - they remain in RS
                        continue
                    
                    # For non-singleton clusters, create CS clusters
                    # Create new CS cluster ID to avoid conflicts
                    new_cs_id = max(list(cs_dict.keys()) + [-1]) + 1
                    
                    # Calculate cluster statistics
                    features = rs_points[point_indices, 2:]
                    n = len(point_indices)
                    SUM = np.sum(features, axis=0)
                    SUMSQ = np.sum(np.square(features), axis=0)
                    
                    # Store statistics
                    cs_dict[new_cs_id] = [n, SUM, SUMSQ]
                    
                    # Calculate centroid and standard deviation
                    centroid = SUM / n
                    std_dev = np.sqrt(np.subtract(SUMSQ / n, np.square(centroid)))
                    
                    cs_centroid_dict[new_cs_id] = centroid
                    cs_std_dev_dict[new_cs_id] = std_dev
                    
                    # Store point IDs and remove from RS
                    point_ids = rs_points[point_indices, 0].astype(int).tolist()
                    cs_point_dict[new_cs_id] = point_ids
                    
                    # Remove these points from RS
                    for point_id in point_ids:
                        if point_id in rs:
                            rs.remove(point_id)
        
        # STEP 12: Merge close CS clusters
        # Identify CS clusters to merge
        cs_to_merge = {}
        
        # First, identify pairs of CS clusters that should be merged
        cs_ids = list(cs_dict.keys())
        for i in range(len(cs_ids)):
            for j in range(i + 1, len(cs_ids)):
                cs_id1 = cs_ids[i]
                cs_id2 = cs_ids[j]
                
                # Skip if either cluster has already been marked for merging
                if cs_id1 in cs_to_merge or cs_id2 in cs_to_merge:
                    continue
                
                # Calculate Mahalanobis distance between centroids
                dist1 = compute_mahalanobis_distance(
                    cs_centroid_dict[cs_id1],
                    cs_centroid_dict[cs_id2],
                    cs_std_dev_dict[cs_id2]
                )
                
                dist2 = compute_mahalanobis_distance(
                    cs_centroid_dict[cs_id2],
                    cs_centroid_dict[cs_id1],
                    cs_std_dev_dict[cs_id1]
                )
                
                min_dist = min(dist1, dist2)
                
                # If close enough, mark for merging
                if min_dist < mahala_threshold:
                    cs_to_merge[cs_id1] = cs_id2
        
        # Perform merging of CS clusters
        for cs_id1, cs_id2 in list(cs_to_merge.items()):
            # Skip if either cluster has been removed already
            if cs_id1 not in cs_dict or cs_id2 not in cs_dict:
                continue
            
            # Merge cs_id1 into cs_id2
            n1, SUM1, SUMSQ1 = cs_dict[cs_id1]
            n2, SUM2, SUMSQ2 = cs_dict[cs_id2]
            
            n_merged, SUM_merged, SUMSQ_merged, centroid_merged, std_dev_merged = merge_clusters(
                n1, SUM1, SUMSQ1, n2, SUM2, SUMSQ2
            )
            
            # Update target cluster
            cs_dict[cs_id2] = [n_merged, SUM_merged, SUMSQ_merged]
            cs_centroid_dict[cs_id2] = centroid_merged
            cs_std_dev_dict[cs_id2] = std_dev_merged
            cs_point_dict[cs_id2].extend(cs_point_dict[cs_id1])
            
            # Remove source cluster
            del cs_dict[cs_id1]
            del cs_centroid_dict[cs_id1]
            del cs_std_dev_dict[cs_id1]
            del cs_point_dict[cs_id1]
        
        # STEP 13: For last round, merge CS with DS clusters
        if round_num == 5:
            cs_to_ds_merge = {}
            
            # Identify CS clusters to merge with DS clusters
            for cs_id in list(cs_dict.keys()):
                min_distance = float('inf')
                nearest_ds = None
                
                for ds_id in ds_dict:
                    # Calculate Mahalanobis distance between centroids
                    dist1 = compute_mahalanobis_distance(
                        cs_centroid_dict[cs_id],
                        ds_centroid_dict[ds_id],
                        ds_std_dev_dict[ds_id]
                    )
                    
                    dist2 = compute_mahalanobis_distance(
                        ds_centroid_dict[ds_id],
                        cs_centroid_dict[cs_id],
                        cs_std_dev_dict[cs_id]
                    )
                    
                    min_dist = min(dist1, dist2)
                    
                    if min_dist < min_distance:
                        min_distance = min_dist
                        nearest_ds = ds_id
                
                # If close enough to a DS cluster, mark for merging
                if min_distance < mahala_threshold and nearest_ds is not None:
                    cs_to_ds_merge[cs_id] = nearest_ds
            
            # Perform merging of CS clusters into DS clusters
            for cs_id, ds_id in cs_to_ds_merge.items():
                # Skip if CS cluster has been removed already
                if cs_id not in cs_dict:
                    continue
                
                # Merge CS into DS
                n_cs, SUM_cs, SUMSQ_cs = cs_dict[cs_id]
                n_ds, SUM_ds, SUMSQ_ds = ds_dict[ds_id]
                
                n_merged, SUM_merged, SUMSQ_merged, centroid_merged, std_dev_merged = merge_clusters(
                    n_cs, SUM_cs, SUMSQ_cs, n_ds, SUM_ds, SUMSQ_ds
                )
                
                # Update DS cluster
                ds_dict[ds_id] = [n_merged, SUM_merged, SUMSQ_merged]
                ds_centroid_dict[ds_id] = centroid_merged
                ds_std_dev_dict[ds_id] = std_dev_merged
                ds_point_dict[ds_id].extend(cs_point_dict[cs_id])
                
                # Remove CS cluster
                del cs_dict[cs_id]
                del cs_centroid_dict[cs_id]
                del cs_std_dev_dict[cs_id]
                del cs_point_dict[cs_id]
        
        # Write round results to output file
        with open(output_path, "a") as f:
            # Count points in each set
            num_ds_points = sum(stats[0] for stats in ds_dict.values())
            num_cs_clusters = len(cs_dict)
            num_cs_points = sum(stats[0] for stats in cs_dict.values() if cs_dict)
            num_rs_points = len(rs)
            
            result_line = f'Round {round_num}: {num_ds_points},{num_cs_clusters},{num_cs_points},{num_rs_points}\n'
            f.write(result_line)
    
    # Prepare final clustering results
    # All points in DS clusters are assigned to their cluster
    # All points in CS clusters and RS are considered outliers (-1)
    result_dict = {}
    
    # Assign DS points to their clusters
    for cluster_id, point_ids in ds_point_dict.items():
        for point_id in point_ids:
            result_dict[point_id] = cluster_id
    
    # Assign CS and RS points as outliers
    for cluster_id, point_ids in cs_point_dict.items():
        for point_id in point_ids:
            result_dict[point_id] = -1
    
    for point_id in rs:
        result_dict[point_id] = -1
    
    # Write final clustering results to the output file
    with open(output_path, "a") as f:
        f.write('\n')
        f.write('The clustering results:\n')
        
        # Sort by point ID
        sorted_point_ids = sorted(result_dict.keys(), key=int)
        for point_id in sorted_point_ids:
            f.write(f'{point_id},{result_dict[point_id]}\n')
    
    # Calculate and print execution time
    end_time = time.time()
    duration = end_time - start_time
    print('Duration:', duration)
    
    return duration

if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python task.py <input_file> <n_cluster> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_path = sys.argv[3]
    
    # Run BFR clustering
    duration = cluster_with_bfr(input_path, n_cluster, output_path)
    
    # Print execution metrics
    print(f"BFR Clustering completed in {duration:.2f} seconds")
    # print(f"Results written to {output_path}")