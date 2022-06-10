#Author: Nand Chandravadia
#email: ndc2136@columbia.edu

import numpy as np


def generate_patterns(n, dim, sparsity):

    patterns = np.zeros(shape = (dim, n))  # each column is a pattern

    for index in range(0, n):
        pattern = generate_binary_vector(dim, sparsity)

        patterns[:, index] = pattern


    return patterns


def generate_binary_vector(dim, sparsity):
    # here, sparsity is defined by the number of (-1)
    # i.e., more (-1), more 'sparse'

    values = [1, -1]

    vec = np.zeros(shape=dim)

    for index, val in enumerate(vec):
        ind = np.random.binomial(1, sparsity)
        vec[index] = values[ind]

    return vec


def compute_orthogonality(length, vec1, vec2):
    return (1 / length) * (vec1.T @ vec2)


def compute_projection_operator(vec):
    m = len(vec)
    projection = np.zeros(shape=(m, m))

    for row_idx, row in enumerate(projection):
        for col_idx, col in enumerate(row):
            projection[row_idx, col_idx] = (1 / m) * (vec[row_idx] * vec[col_idx])

    return projection


def compute_projection_operator_all(patterns):
    # store projections
    projections = {}

    number_patterns = patterns.shape[1]
    dim = patterns.shape[0]

    for index in range(0, number_patterns):
        vec = patterns[:, index]
        vec_proj = compute_projection_operator(vec)

        projections[index] = vec_proj

    return projections

