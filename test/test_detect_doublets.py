"""Tests for detect doublets module.

By: Austin Hartman
Created: 11/26/2020
Last modified: 11/27/2020
"""

import unittest
import scipy.io

import detect_doublets

# load up files
mtx = "/Users/austinhartman/Desktop/bioinfo-analysis/detect_doublets/filtered_feature_bc_matrix/matrix.mtx.gz"
features = "/Users/austinhartman/Desktop/bioinfo-analysis/detect_doublets/filtered_feature_bc_matrix/features.tsv.gz"
barcodes = "/Users/austinhartman/Desktop/bioinfo-analysis/detect_doublets/filtered_feature_bc_matrix/barcodes.tsv.gz"
matrix_dict = detect_doublets.load_feature_barcode_matrix(mtx, barcodes, features)

doublet_finder = detect_doublets.DoubletFinder(matrix_dict["mtx"], matrix_dict["barcodes"])
doublet_finder.find_doublets()

class TestSum(unittest.TestCase):
    """Unit test class.
    """
    def test_csc_matrix_row_addition(self):
        """Test adding sparse rows together works.
        """
        matrix = [[1, 2, 3, 0, 0, 1], [2, 0, 0, 0, 1, 0]]
        matrix = scipy.sparse.csc_matrix(matrix)
        doublet_obj = detect_doublets.DoubletFinder(matrix)

        assert (
            (doublet_obj.mtx.getrow(0) + doublet_obj.mtx.getrow(1)).toarray()[0] == [3, 2, 3, 0, 1, 1]
        ).all()

    def test_doublet_obj_shape(self):
        """Test shape of sparsified matrix.
        """
        matrix = [[1, 2, 3, 0, 0, 1], [2, 0, 0, 0, 1, 0]]
        matrix = scipy.sparse.csc_matrix(matrix)
        doublet_obj = detect_doublets.DoubletFinder(matrix)
        assert doublet_obj.mtx.shape == (2, 6)

    def test_doublet_obj_create_doublet(self):
        """Test that creation of artificial doublets add the expected number of rows.
        """
        matrix = [
            [2, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1],
        ]
        matrix = scipy.sparse.csc_matrix(matrix)
        doublet_object = detect_doublets.DoubletFinder(matrix, artificial_fraction=0.75)
        doublet_object._create_artificial_doublets() # pylint: disable=protected-access
        assert (doublet_object.mtx.toarray()[:, 6] == [4, 0, 6, 2]).all()

    def test_similarity_indices_for_cell_barcodes_only(self):
        """verifying stuff that isn't a cell doesn't get counted.
        """
        for i, _ in doublet_finder.num_times_knn:
            assert i < 8499 # 8,499 is the number of cells called in the dataset

    def test_all_similarity_indices_present(self):
        """making sure all cell barcodes are found in the similarity metric.
        """
        check_present = [False for i in range(8499)]
        for i, _ in doublet_finder.num_times_knn:
            check_present[i] = True
        assert sum(check_present) == 8499


if __name__ == "__main__":
    unittest.main()
