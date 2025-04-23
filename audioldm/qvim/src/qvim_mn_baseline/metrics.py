import pandas as pd
import numpy as np


def compute_mrr(similarity_df: pd.DataFrame, ground_truth: dict) -> float:
    reciprocal_ranks = []

    similarity_df = similarity_df[~similarity_df.index.duplicated(keep='first')]

    for query in similarity_df.index:
        if query in ground_truth:
            correct_item = ground_truth[query]

            # Sort items by similarity score in descending order
            ranked_items = similarity_df.loc[query].sort_values(ascending=False)

            # Get the rank (1-based index)
            if correct_item in ranked_items.index:
                rank = ranked_items.index.get_loc(correct_item) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                print(query, correct_item)
                reciprocal_ranks.append(0)

    # Compute Mean Reciprocal Rank (MRR)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0


def compute_ndcg(similarity_df: pd.DataFrame, ground_truth: dict) -> float:
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) for ranked retrieval results.

    similarity_df: DataFrame with queries as index and items as columns.
    ground_truth: Dictionary mapping queries to a set of relevant items with relevance scores.
    """
    ndcg_scores = []

    similarity_df = similarity_df[~similarity_df.index.duplicated(keep='first')]

    for query in similarity_df.index:
        if query in ground_truth:
            relevant_items = ground_truth[query]  # Dictionary {item: relevance_score}

            # Sort retrieved items by similarity score in descending order
            ranked_items = similarity_df.loc[query].sort_values(ascending=False)

            # Compute DCG for retrieved ranking
            retrieved_relevance_scores = [int(item in relevant_items) for item in ranked_items.index]
            dcg = dcg_at(retrieved_relevance_scores)

            # Compute ideal DCG (IDCG) for perfect ranking
            ideal_relevance_scores = [1]*len(relevant_items) + [0]*(len(retrieved_relevance_scores) - len(relevant_items))
            idcg = dcg_at(ideal_relevance_scores)

            # Compute NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0



def dcg_at(scores):
    """Compute Discounted Cumulative Gain (DCG) at rank K."""
    scores = np.array(scores)
    return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))