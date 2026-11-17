"""Indexing for finite tensor products with ordered factors."""


def _basis_indices(dimensions, permutation, total):
    """Map each permuted basis index to its source index in lexicographic order.

    The caller validates dimensions, permutation and total matrix size.
    """
    target_dimensions = tuple(dimensions[p] for p in permutation)
    indices = []
    for new_index in range(total):
        remainder, source_coordinates = new_index, [0]*len(dimensions)
        for target_party in range(len(dimensions)-1, -1, -1):
            remainder, coordinate = divmod(remainder, target_dimensions[target_party])
            source_coordinates[permutation[target_party]] = coordinate
        source_index = 0
        for coordinate, dimension in zip(source_coordinates, dimensions):
            source_index = source_index*dimension + coordinate
        indices.append(source_index)
    return tuple(indices)
