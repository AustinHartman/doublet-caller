"""Provides class for not-very-efficient calculation of nearest neighbors
for some index in a matrix.

By: Austin Hartman
Created: 11/26/2020
Last modified: 11/27/2020
"""

from numpy import genfromtxt


def main():
    """load up the PCA matrix and find the nearest neighbors.
    """
    matrix = genfromtxt(
        "/Users/austinhartman/Desktop/bioinfo-analysis/detect_doublets/pca_matrix.csv",
        delimiter=",",
    )

    knn = NearestNeighbors(matrix, 3)

    print(knn.get_nearest_neighbors(3))


class NearestNeighbors:
    """class to calculate which cells are most similar to another.
    The class is not efficient, but does the trick when calculating
    similarity on a small number of gene expression profiles in a
    relatively small matrix.
    """

    def __init__(self, mtx, k, distance="euclidean"):
        self.mtx = mtx
        self.k = k
        self.distance = distance
        self.similarity_matrix = None

    def get_nearest_neighbors(self, idx, idxs_to_ignore=None):
        """
        return the nearest neighbors for a given index based on the similarity matrix.
        """

        # if self.similarity_matrix is None:
        #    sys.exit("Please run calculate_similarity_matrix() first.")

        sim_list = list()  # keep track of the similarities

        # each iteration adds a tuple: (similarity score, index)
        for i in range(len(self.mtx)):
            if idxs_to_ignore and i in idxs_to_ignore:
                pass
            else:
                sim_list.append((euclidean(self.mtx[i], self.mtx[idx]), i))

        # return the k most similar cell profiles
        return sorted(sim_list)[1 : self.k + 1]


def euclidean(ls_1, ls_2):
    """Calculate the euclidean distance between two lists.
    """
    return sum([(j - k) ** 2 for j, k in zip(ls_1, ls_2)]) ** 0.5


if __name__ == "__main__":
    main()
