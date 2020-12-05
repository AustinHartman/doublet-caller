"""Tests for nearest neighbors module.

By: Austin Hartman
Created: 11/27/2020
Last modified: 11/27/2020
"""

import unittest
import math

import nearest_neighbors

class TestSum(unittest.TestCase):
    """Test class for nearest neighbors module.
    """

    def test_euclidean_distance(self):
        """Check correctness of euclidean distance function.
        """
        ls_1 = [8, 6]
        ls_2 = [5, 1]
        print(nearest_neighbors.euclidean(ls_1, ls_2))
        assert nearest_neighbors.euclidean(ls_1, ls_2) == math.sqrt(34)

    def test_correct_nearest_neighbors(self):
        """Test that correct number of nearest neighbors is returned.
        """
        mtx = [
            [1, 2, 3, 4],
            [2, 1, 3, 4],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ]

        nn_object = nearest_neighbors.NearestNeighbors(mtx, 3)
        assert len(nn_object.get_nearest_neighbors(2)) == 3


if __name__ == "__main__":
    unittest.main()
