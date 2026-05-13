import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import fisher_exact


def generate_pairwise_similarity_matrix(
    aggregated_features: pd.DataFrame,
    metadata: pd.DataFrame
) -> pd.DataFrame:
    # Compute the full pairwise cosine similarity matrix
    similarity_matrix = cosine_similarity(aggregated_features)

    #
    n_samples = len(aggregated_features)
    upper_tri_indices = np.triu_indices(n_samples, k=1)  # k=1 excludes diagonal
    similarities = similarity_matrix[upper_tri_indices]

    # Create arrays for indices
    indices_i = upper_tri_indices[0]
    indices_j = upper_tri_indices[1]

    # Get gene symbols efficiently
    gene_symbols = metadata["Metadata_Symbol"].values
    sample1_symbols = gene_symbols[indices_i]
    sample2_symbols = gene_symbols[indices_j]

    # Create DataFrame directly with vectorized operations
    pairwise_similarity_df = pd.DataFrame({
        "similarity": similarities,
        "index1": indices_i,
        "index2": indices_j,
        "sample1_Metadata_Symbol": sample1_symbols,
        "sample2_Metadata_Symbol": sample2_symbols
    })

    return pairwise_similarity_df

def generate_pairwise_sim_with_stringdb(
    pairwise_similarity_df: pd.DataFrame,
    protein_interactions: pd.DataFrame
) -> pd.DataFrame:
    pairwise_similarity_df = pd.merge(
        pairwise_similarity_df,
        protein_interactions,
        left_on=["sample1_Metadata_Symbol", "sample2_Metadata_Symbol"],
        right_on=["protein1_symbol", "protein2_symbol"],
        how="left"
    )
    pairwise_similarity_df = pairwise_similarity_df.dropna(axis=0)
    pairwise_similarity_df = pairwise_similarity_df.sort_values("similarity", ascending=False).reset_index(drop=True)
    pairwise_similarity_df.drop(columns=["index1", "index2", "protein1_symbol", "protein2_symbol"], inplace=True)
    return pairwise_similarity_df

def get_similar_dissimilar_pairs(
    pairwise_similarity_df: pd.DataFrame,
    percentage: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame]:
    similar_5_pct = pairwise_similarity_df.head(int(len(pairwise_similarity_df) * percentage))
    dissimilar_5_pct = pairwise_similarity_df.tail(int(len(pairwise_similarity_df) * percentage))

    background =  pairwise_similarity_df.tail(int(len(pairwise_similarity_df) * (1 - percentage)))
    return similar_5_pct, dissimilar_5_pct, background

def p_val_fisher_exact(target_sample_size, target_size, background_sample_size, background_size):
    table = np.array([[target_sample_size, target_size],
                    [background_sample_size, background_size]])

    odds_ratio, p_value = fisher_exact(table, alternative='greater')
    return odds_ratio, p_value

def load_features(features_path: str):
    with open(features_path, "rb") as f:
        normalized_features = pickle.load(f)
    return normalized_features

def filter_features(features):
    filtered_features = {}
    for key, value in features.items():
        feature_cols = [col for col in value.columns if col.startswith("PC")]
        filtered_features[key] = value[["Metadata_Well", "Metadata_Batch", "Metadata_JCP2022", "Metadata_Symbol"] + feature_cols]
    return filtered_features

def aggregate_features_tgt2(features):
    aggregated_features = {}
    for key, value in features.items():
        feature_cols = [col for col in value.columns if col.startswith("PC")]
        aggregated = value.groupby(["Metadata_JCP2022", "Metadata_Symbol"])[feature_cols].mean().reset_index()
        aggregated_features[key] = aggregated
    return aggregated_features

# find a better way to deal with feature columns
def aggregate_features(features):
    aggregated_features = {}
    for key, value in features.items():
        feature_cols = [col for col in value.columns if col.startswith("PC")]
        aggregated = value.groupby("Metadata_Symbol")[feature_cols].mean().reset_index()
        aggregated_features[key] = aggregated
    return aggregated_features

def generate_pairwise_similarity(aggregated_features):
    pairwise_similarity = {}

    metadata = aggregated_features["dino_v2"][["Metadata_Symbol"]]
    feature_cols = [col for col in aggregated_features["dino_v2"].columns if col.startswith("PC")]

    for key, value in aggregated_features.items():
        similarity_df = generate_pairwise_similarity_matrix(
            aggregated_features=value[feature_cols].values, 
            metadata=metadata)
        pairwise_similarity[key] = similarity_df

    return pairwise_similarity

def map_stringdb_clusters(pairwise_similarity, stringdb_interactions):
    features_with_stringdb = {}
    for key, value in pairwise_similarity.items():
        features_with_stringdb[key] = generate_pairwise_sim_with_stringdb(value, stringdb_interactions)
    return features_with_stringdb

def filter_clusters():
    pass

def get_pairs_of_interest(pairwise_similarity, cutoff=0.025, double_cutoff=True):
    pairs_of_interest = {}
    background_pairs_dict = {}
    for key, value in pairwise_similarity.items():
        similar_pairs, dissimilar_pairs, background_pairs = get_similar_dissimilar_pairs(value, cutoff)

        combined_pairs = pd.concat([similar_pairs, dissimilar_pairs])
        combined_pairs.reset_index(drop=True, inplace=True)

        if double_cutoff:
            pairs_of_interest[key] = combined_pairs
            background_pairs_dict[key] = background_pairs.head(int(len(pairwise_similarity[key]) * (1-cutoff)))
        else:
            background_pairs_dict[key] = background_pairs
            pairs_of_interest[key] = similar_pairs

    return pairs_of_interest, background_pairs_dict

def get_odds_ratio_and_pval_compounds(pairs_of_interest, background_pairs):

    target_count = (pairs_of_interest['dino_v2']["sample1_Metadata_Symbol"] == pairs_of_interest['dino_v2']["sample2_Metadata_Symbol"]).sum()
    background_count = (background_pairs['dino_v2']["sample1_Metadata_Symbol"] == background_pairs['dino_v2']["sample2_Metadata_Symbol"]).sum()

    results = {}


    for key, value in pairs_of_interest.items():
        print(target_count, len(value), background_count, len(background_pairs[key]))
        odds_ratio, pval = p_val_fisher_exact(
            target_sample_size=target_count,
            target_size=len(value),
            background_sample_size=background_count,
            background_size=len(background_pairs[key])
        )
        results[key] = {
            "odds_ratio": odds_ratio,
            "pval": pval
        }

    return results

def get_odds_ratio_and_pval(pairs_of_interest, background_pairs):

    results = {}
    for key, value in pairs_of_interest.items():
        odds_ratio, pval = p_val_fisher_exact(
            target_sample_size=np.sum(value["combined_score"] > 950),
            target_size=len(value),
            background_sample_size=np.sum(background_pairs[key]["combined_score"] > 950),
            background_size=len(background_pairs[key])
        )
        results[key] = {
            "odds_ratio": odds_ratio,
            "pval": pval
        }

    return results

def map_cluster_id(features, stringdb_clusters_path):
    stringdb_clusters = pd.read_pickle(stringdb_clusters_path)
    stringdb_clusters = stringdb_clusters[~stringdb_clusters["protein_symbol"].str.contains("ENSP")]

    stringdb_clusters_dict = dict(zip(stringdb_clusters['protein_symbol'], stringdb_clusters['cluster_id']))
    for key, value in features.items():
        features[key]["protein1_cluster"] = features[key]["sample1_Metadata_Symbol"].map(stringdb_clusters_dict)
        features[key]["protein2_cluster"] = features[key]["sample2_Metadata_Symbol"].map(stringdb_clusters_dict)
    return features


def get_filtered_clusters(pairwise_similarity, min_occurrences=5, max_occurrences=200):
    untrained_resnet_with_stringdb = pairwise_similarity["dino_v2"]
    same_cluster_pairs = untrained_resnet_with_stringdb[
        (untrained_resnet_with_stringdb["protein1_cluster"] == untrained_resnet_with_stringdb["protein2_cluster"]) &
        (untrained_resnet_with_stringdb["protein1_cluster"].notna()) &
        (untrained_resnet_with_stringdb["protein2_cluster"].notna())
    ]

    # Count occurrences of each cluster_id in one operation
    cluster_count = same_cluster_pairs["protein1_cluster"].value_counts().to_dict()

    # Filter clusters with more than min_occurrences
    filtered_clusters = {key: value for key, value in cluster_count.items() if (value > min_occurrences and value < max_occurrences)}
    return filtered_clusters

def plot_filtered_cluster(pairwise_similarity, start=0, end=100, step=5):
    min_occurrences_list = list(range(start, end, step))
    num_clusters = []

    for min_occ in min_occurrences_list:
        filtered = get_filtered_clusters(pairwise_similarity, min_occurrences=min_occ)
        num_clusters.append(len(filtered))

    plt.figure(figsize=(10, 6))
    plt.plot(min_occurrences_list, num_clusters, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Minimum Occurrences', fontsize=12)
    plt.ylabel('Number of Clusters', fontsize=12)
    plt.title('Number of Clusters vs Minimum Occurrences Threshold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()

def get_weighted_odds_ratios(pairs_of_interest, background_pairs, filtered_clusters):
    weighted_odds_ratios = {}
    for key in pairs_of_interest.keys():
        weighted_odds_ratios[key] = {}
        for cluster_id in tqdm(list(filtered_clusters.keys())):
            cluster_id = str(cluster_id)
            matching_clusters_target = len(pairs_of_interest[key][((pairs_of_interest[key]["protein1_cluster"] == cluster_id) & (pairs_of_interest[key]["protein2_cluster"] == cluster_id))
                                                                  & (pairs_of_interest[key]["combined_score"] >= 950)])
            mismatching_clusters_target = len(pairs_of_interest[key][((pairs_of_interest[key]["protein1_cluster"] == cluster_id) | (pairs_of_interest[key]["protein2_cluster"] == cluster_id))
                                                                    & (pairs_of_interest[key]["combined_score"] < 950)])
            matching_clusters_background = len(background_pairs[key][((background_pairs[key]["protein1_cluster"] == cluster_id) & (background_pairs[key]["protein2_cluster"] == cluster_id))
                                                                     & (background_pairs[key]["combined_score"] >= 950)])
            mismatching_clusters_background = len(background_pairs[key][((background_pairs[key]["protein1_cluster"] == cluster_id) | (background_pairs[key]["protein2_cluster"] == cluster_id))
                                                                       & (background_pairs[key]["combined_score"] < 950)])
            odds, p_val = p_val_fisher_exact(target_sample_size=matching_clusters_target, target_size=mismatching_clusters_target,
                                            background_sample_size=matching_clusters_background, background_size=mismatching_clusters_background)
            if np.isnan(odds):
                odds = 0
            if np.isinf(odds):
                odds = len(mismatching_clusters_background)
            weighted_odds_ratios[key][cluster_id] = odds
    return weighted_odds_ratios

def get_odds_ratio_per_compound(pairs_of_interest, background_pairs):
    odds_ratios = {}
    for key in pairs_of_interest.keys():
        odds_ratios[key] = {}
        matching_clusters_target = len(pairs_of_interest[key][(pairs_of_interest[key]["combined_score"] >= 950)])
        mismatching_clusters_target = len(pairs_of_interest[key][(pairs_of_interest[key]["combined_score"] < 950)])
        matching_clusters_background = len(background_pairs[key][(background_pairs[key]["combined_score"] >= 950)])
        mismatching_clusters_background = len(background_pairs[key][(background_pairs[key]["combined_score"] < 950)])
        odds, p_val = p_val_fisher_exact(target_sample_size=matching_clusters_target, target_size=mismatching_clusters_target,
                                        background_sample_size=matching_clusters_background, background_size=mismatching_clusters_background)
        if np.isnan(odds):
            odds = 0
        if np.isinf(odds):
            odds = len(mismatching_clusters_background)
        odds_ratios[key] = odds
    return odds_ratios
