import torch


def low_rank_decomposition(weight_matrix, rank):
    # Perform Singular Value Decomposition (SVD)
    U, S, V = torch.svd(weight_matrix)

    # Keep only the top 'rank' singular values and vectors
    U = U[:, :rank]
    S = S[:rank]
    V = V[:, :rank]

    # Reconstruct the low-rank approximation
    W_approx = torch.mm(U, torch.mm(torch.diag(S), V.t()))

    return W_approx


# Example usage
# Create a random weight matrix of size 6x4
original_weight = torch.randn(6, 4)

# Specify the rank for decomposition
rank = 2

# Apply low-rank decomposition
approximated_weight = low_rank_decomposition(original_weight, rank)

# Display the results
print("Original Weight Matrix:\n", original_weight)
print("\nApproximated Weight Matrix (Rank = {}):\n".format(rank), approximated_weight)
