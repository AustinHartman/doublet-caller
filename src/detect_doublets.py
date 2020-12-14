"""Module for calling doublets from scRNA-seq or scATAC-seq data.

By: Austin Hartman
Created: 11/~/2020
Last modified: 12/5/2020
"""

import csv
import gzip
import sys
import argparse
import collections
import scipy  # io package requires it's own explicit import
import scipy.io
import numpy as np
from sklearn import decomposition

import nearest_neighbors

RANDOM_STATE = 24
ARTIFICIAL_FRACTION_DEFAULT = 0.03
NUM_TO_SAVE_AS_DOUBLETS_DEFAULT = 100

def main():
    """Run the module"""

    # parse arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--features", help="path to features.tsv.gz", required=True)
    parser.add_argument("--matrix", help="path to matrix.mtx.gz", required=True)
    parser.add_argument("--barcodes", help="path to barcodes.tsv.gz", required=True)
    parser.add_argument(
        "--doublet-file",
        help="file path to save putative doublets",
        default=None
    )
    parser.add_argument(
        "--artificial-fraction",
        help="Number of artificial doublets to generate as a fraction of the number of cells",
        default=ARTIFICIAL_FRACTION_DEFAULT
    )
    args = parser.parse_args()

    # load up files
    # mtx = "/Users/austinhartman/Desktop/bioinfo-analysis/detect_doublets/filtered_feature_bc_matrix/matrix.mtx.gz"
    # features = "/Users/austinhartman/Desktop/bioinfo-analysis/detect_doublets/filtered_feature_bc_matrix/features.tsv.gz"
    # barcodes = "/Users/austinhartman/Desktop/bioinfo-analysis/detect_doublets/filtered_feature_bc_matrix/barcodes.tsv.gz"

    matrix_dict = load_feature_barcode_matrix(args.matrix, args.barcodes, args.features)

    doublet_finder = DoubletFinder(matrix_dict["mtx"], matrix_dict["barcodes"])
    doublet_finder.find_doublets(save_barcodes_path=args.doublet_file)
    doublet_finder.print_metrics()

    print("All done!")


class DoubletFinder:
    """Class which takes feature barcode matrix and calls doublet barcodes"""

    def __init__(
            self,
            mtx,
            barcodes=None,
            feature_ids=None,
            feature_types=None,
            gene_names=None,
            artificial_fraction=ARTIFICIAL_FRACTION_DEFAULT
    ):
        # self.matrix_dict = _load_market_mtx(mtx, barcodes, features)
        # Required
        self.mtx = mtx

        # Optional
        self.barcodes = barcodes
        self.feature_ids = feature_ids
        self.feature_types = feature_types
        self.gene_names = gene_names
        self.artificial_fraction = artificial_fraction

        # Calculated based on other args
        self.num_cells = self.mtx.get_shape()[1]

        # the index of this list represents the cell and the value represents the number of times its labelled a k-nn
        self.num_times_knn = [[i, 0] for i in range(self.num_cells)]
        self.num_genes = self.mtx.get_shape()[0]
        self.pca_matrix = np.ndarray([])
        self.nearest_neighbors_dict = dict()
        self.num_cells_for_artifial_doublets, self.num_artifial_doublets = self._calc_num_artifical()
        self.doublet_barcodes = list() # store the doublet barcodes which can be later saved as a TSV

    def find_doublets(
            self,
            k=15,
            save_pca_path=None,
            save_mtx_path=None,
            save_barcodes_path=None
    ):
        """Function that wires it all together and calls doublets."""

        if save_mtx_path:
            self._save_matrix(save_mtx_path)
        self._create_artificial_doublets()
        self._reduce_matrix_dimensions()
        if save_pca_path:
            self._save_pca_matrix(save_pca_path)
        self._find_nearest_neighbors(k)
        self._call_doublets()
        if save_barcodes_path:
            self._save_barcodes(save_barcodes_path)

    def _calc_num_artifical(self):
        # set number of cells to use for artificial doublet generation
        n = int(self.artificial_fraction * self.num_cells)
        num_cells_for_artifial_doublets = n if n % 2 == 0 else n - 1
        assert num_cells_for_artifial_doublets % 2 == 0

        # index 0 is the number of cells used to generate artificial doublets
        # index 1 is the number of artificial doublets to be created
        return (num_cells_for_artifial_doublets, int(num_cells_for_artifial_doublets / 2))

    def _create_artificial_doublets(self):
        """
        in order to create doublets, select two random gene expression profiles
        and combine them into one artificial profile. Repeat this process for
        some reasonable number.
        """

        # set numpy seed
        np.random.seed(RANDOM_STATE)

        # generate list of ints for each cell column idx
        cells = [i for i in range(self.num_cells)]

        # randomly sample artificial_fraction% of the cells to be used in doublet generation
        cells_for_artifical_doublets = np.random.choice(
            cells, self.num_cells_for_artifial_doublets, replace=False
        )

        # TODO: convert this data structure into a tuple of tuples of len 2 representing the two cell

        # verify selection was done without replacement
        assert len(cells_for_artifical_doublets) == len(set(cells_for_artifical_doublets))

        # fill in artificial doublet matrix
        extension_mtx = np.zeros(shape=(self.num_genes, self.num_artifial_doublets))
        for i in range(0, len(cells_for_artifical_doublets), 2):
            arr = (
                self.mtx.getcol(cells_for_artifical_doublets[i]).toarray()
                + self.mtx.getcol(cells_for_artifical_doublets[i + 1]).toarray()
            )
            arr.shape = (arr.shape[0],)
            extension_mtx[:, int(i / 2)] = arr

        # make the numpy array into sparse scipay matrix
        extension_mtx = scipy.sparse.csc_matrix(extension_mtx)

        # append extension_mtx to existing sparse matrix
        self.mtx = scipy.sparse.hstack((self.mtx, extension_mtx))

    def _reduce_matrix_dimensions(self, principal_components=5):
        """
        take the matrix and run PCA on it so that it is faster to compute
        similarity measures to the simulated doublets downstream.
        """
        # TODO: consider the TruncatedSVD or other sklearn PCA algorithm varients here

        pca = decomposition.PCA(n_components=principal_components, random_state=RANDOM_STATE)
        pca.fit(np.rot90(self.mtx.toarray())) # rotate by 90 degrees to accomadate for axis which reduction is performed on
        self.pca_matrix = pca.transform(np.rot90(self.mtx.toarray()))

    def _find_nearest_neighbors(self, k=15):
        """
        Calculate a similarity matrix based on the reduced dimension PCA matrix.
        Experiment with different similarity metrics, but may need to make a decision
        partly on the basis of speed at this step.
        """
        # this isn't running as expected
        # if self.pca_matrix.any():
        #     sys.exit("Please run reduce matrix dimensions for populate the PCA matrix.")

        # key will represent index for artificial doublet
        # value will hold list of the most similar doublets
        nn_obj = nearest_neighbors.NearestNeighbors(self.pca_matrix, k)

        # create set of indices for nearest neighbors to ignore; set contains indices for artificial doublets
        idxs_to_ignore = {i for i in range(self.num_cells, self.num_cells + self.num_artifial_doublets)}
        for i in range(self.num_cells, self.num_cells+self.num_artifial_doublets):
            neighbors = nn_obj.get_nearest_neighbors(i, idxs_to_ignore)
            neighbors = [i for i in neighbors if i[1] < self.num_cells]  # only include similarity if that similarity is for a cell barcode
            self.nearest_neighbors_dict[i] = neighbors

    def _call_doublets(self):
        """
        will need to do some experimentation here. I would like to see if
        a gaussian mixture model can be fit to the similarities to each
        simulated doublet. Not sure if this will look like a bimodal distribution
        or not.
        """

        # look through the nearest_neighbors_dict to find  cell barcodes
        # which are regularly marked as similar to artificial doublets
        for _, v in self.nearest_neighbors_dict.items():
            for _, cell_idx in v:
                self.num_times_knn[cell_idx][1] += 1

        self.doublet_barcodes = sorted(self.num_times_knn, key=lambda x: x[1])[-NUM_TO_SAVE_AS_DOUBLETS_DEFAULT:]
        # print(sorted(self.num_times_knn, key=lambda x: x[1])[-40:])

    def print_metrics(self):
        """Print relevant data for debugging."""
        # num times regular barcodes appear in a simulated doublet nearest neighbors, grouped by value
        # TODO: this list is 2 dimensional... need to extract dimensione with counts for the counter
        frequencies = [i[1] for i in self.num_times_knn]
        counter = collections.Counter(
            frequencies
        )
        print("##\nNumber time barcoded in sim doub KNN: {}".format(counter))

        # artificial fraction
        print("##\nArtificial fraction: {}".format(self.artificial_fraction))


    def _save_barcodes(self, filename, num_to_save_as_doublets=NUM_TO_SAVE_AS_DOUBLETS_DEFAULT):
        # TODO: come up with a better metric to score barcodes. Number of times a barcode
        # appears in a simulated doublets nearest neighbors is likely not all that robust
        with open(filename, 'w') as f:
            for i, _ in sorted(self.num_times_knn, key=lambda x: x[1])[-num_to_save_as_doublets:]:
                f.write("{},\n".format(self.barcodes[i]))

    def _save_matrix(self, filename):
        """Save matrix as market matrix format"""

        scipy.io.mmwrite(filename, self.mtx, field="integer")  # pylint: disable=no-member

    def _save_pca_matrix(self, filename):
        """Save reduced dimension PCA matrix as a CSV."""

        np.savetxt(filename, self.pca_matrix, delimiter=",")


def load_feature_barcode_matrix(mtx, barcodes_path, features_path):
    """Load 10x feature barcode matrix files."""

    mat = scipy.io.mmread(mtx)  # pylint: disable=no-member
    mat = scipy.sparse.csc_matrix(mat)

    try:
        with gzip.open(features_path, "rt") as f:
            feature_ids = [row[0] for row in csv.reader(f, delimiter="\t")]
        with gzip.open(features_path, "rt") as f:
            feature_types = [row[0] for row in csv.reader(f, delimiter="\t")]
        with gzip.open(features_path, "rt") as f:
            gene_names = [row[1] for row in csv.reader(f, delimiter="\t")]
        with gzip.open(barcodes_path, "rt") as f:
            barcodes = [row[0] for row in csv.reader(f, delimiter="\t")]
    except OSError:
        sys.exit("OSError")

    return {
        "feature_ids": feature_ids,
        "feature_types": feature_types,
        "gene_names": gene_names,
        "mtx": mat,
        "barcodes": barcodes,
    }


if __name__ == "__main__":
    main()
