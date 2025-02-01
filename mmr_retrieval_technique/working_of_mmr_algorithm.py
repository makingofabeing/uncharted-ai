def compute_mmr(query, candidate_docs, selected_docs, relevance_scores, similarity_matrix, lambda_param=0.7):
    """
    Implements Maximum Marginal Relevance (MMR).

    Parameters:
        query (str): The user's query (not used directly in this example, relevance scores are precomputed).
        candidate_docs (list): List of candidate documents.
        selected_docs (list): List of already selected documents.
        relevance_scores (dict): Relevance scores for each candidate document to the query.
        similarity_matrix (dict): Similarity scores between all pairs of documents.
        lambda_param (float): Balancing parameter between relevance and diversity (0 <= lambda_param <= 1).

    Returns:
        list: Ordered list of selected documents based on MMR.
    """
    selected = list(selected_docs)  # Copy selected docs
    candidates = [doc for doc in candidate_docs if doc not in selected_docs]

    while candidates:
        mmr_scores = {}

        for candidate in candidates:
            # Calculate relevance score
            rel_score = relevance_scores.get(candidate, 0)

            # Calculate diversity penalty
            if selected:
                sim_scores = [similarity_matrix.get((candidate, s), 0) for s in selected]
                max_sim = max(sim_scores)
            else:
                max_sim = 0  # No diversity penalty if no documents are selected

            # MMR formula
            mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim
            mmr_scores[candidate] = mmr_score

        # Select the document with the highest MMR score
        best_doc = max(mmr_scores, key=mmr_scores.get)
        selected.append(best_doc)
        candidates.remove(best_doc)

    return selected

def main():
    """Let's assume we have 4 documents in the vector database stored as embeddings.
    This is represented as candidate_docs"""
    candidate_docs = ["D1", "D2", "D3", "D4"]

    selected_docs = []  # No documents selected initially

    # Let's assume we have the relevance score to the query pre-computed using metrics such as cosine similarity.
    relevance_scores = {
        "D1": 0.9,
        "D2": 0.8,
        "D3": 0.7,
        "D4": 0.6
    }

    # Let's also assume we have the pairwise document similarity computed
    similarity_matrix = {
        ("D1", "D2"): 0.5, ("D1", "D3"): 0.6, ("D1", "D4"): 0.4,
        ("D2", "D1"): 0.5, ("D2", "D3"): 0.7, ("D2", "D4"): 0.5,
        ("D3", "D1"): 0.6, ("D3", "D2"): 0.7, ("D3", "D4"): 0.4,
        ("D4", "D1"): 0.4, ("D4", "D2"): 0.5, ("D4", "D3"): 0.4,
    }

    # Run MMR
    selected_docs_order = compute_mmr(
        query="Example Query",
        candidate_docs=candidate_docs,
        selected_docs=selected_docs,
        relevance_scores=relevance_scores,
        similarity_matrix=similarity_matrix,
        lambda_param=0.7
    )

    print("Selected Documents in Order:", selected_docs_order)


if __name__ == "__main__":
    main()
