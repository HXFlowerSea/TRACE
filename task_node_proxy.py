import math
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def progressive_clustering(cur_rep_node, embs, node_clu_idx, num_clu, args, random_state=1, parent_id=None, depth=0):
    current_id = f"depth_{depth}_parent_{parent_id}_numclu_{num_clu}" if parent_id else f"depth_{depth}_numclu_{num_clu}"
    cluster_info, count_list, compactness_list, center_node_list = embs_clu(embs, node_clu_idx, num_clu, random_state)

    if check_compactness(compactness_list, args.variance_threshold):
        cur_rep_node += center_node_list
        return cur_rep_node

    if args.compactness_threshold_mode == '28':
        # Determine the compactness threshold using the 80/20 rule (Pareto principle)
        sorted_compactness = sorted(compactness_list, reverse=True)
        threshold_index = max(math.ceil(len(sorted_compactness) * 0.2), 0)

        if threshold_index == 0:
            # Use a fallback threshold (e.g., mean compactness or a predefined max threshold)
            compactness_threshold = min(np.mean(sorted_compactness), max(sorted_compactness))
        else:
            compactness_threshold = sorted_compactness[threshold_index]
    elif args.compactness_threshold_mode == 'avg':
        compactness_threshold = sum(compactness_list) / len(compactness_list)

    # For clusters exceeding the threshold, perform recursive clustering
    for cluster_id in range(len(compactness_list)):
        if compactness_list[cluster_id] > compactness_threshold:
            new_num_clu = round(num_clu * (cluster_info[cluster_id]["node_count"] / float(embs.shape[0])))
            new_num_clu = max(2, new_num_clu)
            if new_num_clu <= 2 and cluster_info[cluster_id]["node_count"] / float(embs.shape[0]) < 0.2:
                cur_rep_node.append(center_node_list[cluster_id])
                continue
            cur_rep_node = progressive_clustering(
                cur_rep_node, 
                embs,
                np.array(cluster_info[cluster_id]["node_indices"]),
                new_num_clu,
                args,
                random_state,
                parent_id=current_id,  
                depth=depth + 1, 
            )
        else:
            cur_rep_node.append(center_node_list[cluster_id])

    return cur_rep_node

def embs_clu(embs, node_clu_idx, num_clu, random_state=1):
    subset_embs = embs[node_clu_idx] 

    # Perform KMeans clustering on the subset
    #kmeans = KMeans(n_clusters=num_clu, random_state=random_state)
    kmeans = KMeans(n_clusters=num_clu)
    pseudo_labels = kmeans.fit_predict(subset_embs)  # Generate pseudo-labels
    cluster_centers = kmeans.cluster_centers_

    # Store information
    cluster_info = {}
    count_list = []
    compactness_list = []
    center_node_list = []

    for cluster_id in range(num_clu):
        # Get indices of nodes belonging to the current cluster (local indices)
        local_indices = np.where(pseudo_labels == cluster_id)[0]

        # Map local indices back to original indices
        original_indices = node_clu_idx[local_indices]
        cluster_embeddings = subset_embs[local_indices]

        # Find the node closest to the cluster center (original index)
        distances = cdist(cluster_embeddings, cluster_centers[cluster_id].reshape(1, -1))
        closest_node_index = original_indices[np.argmin(distances)]
        local_center_node = local_indices[np.argmin(distances)]
        center_node_list.append(closest_node_index)

        # Calculate compactness (mean distance to the cluster center)
        compactness = np.mean(distances)

        # Store cluster information
        cluster_info[cluster_id] = {
            "node_indices": list(original_indices),  # Original indices of the nodes in this cluster
            "local_indices": list(local_indices),  # Local indices relative to the subset
            "node_embeddings": cluster_embeddings,  # Embeddings of the nodes in this cluster
            "pseudo_labels": cluster_id,  # Cluster label
            "node_count": len(local_indices),  # Number of nodes in this cluster
            "compactness": compactness,  # Compactness of the cluster
            "center_node": closest_node_index,  # Center node (original index)
            "local_center_node": local_center_node
        }

        # Append size and compactness for this cluster
        count_list.append(len(local_indices))
        compactness_list.append(compactness)

    return cluster_info, count_list, compactness_list, center_node_list

def check_compactness(compactness_list, threshold=0.5):
    variance = np.var(compactness_list)
    return variance < threshold