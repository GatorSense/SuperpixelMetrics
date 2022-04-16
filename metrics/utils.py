import numpy as np
from typing import List
import numpy.typing as npt
import scipy as sp


def ComputeAdjacency(
    labels: npt.NDArray[np.int_], K: int, connectivity=8
) -> tuple(npt.NDArray[np.int_], List[int]):
    """
    Compute adjacency matrix

    Inputs
    ------
    labels: np.ndarray (int) shape:(image)
        NumPy array containing superpixel label of each pixel.
    K: int
        Number of superpixels within the image.
    connectivity: int [4 | 8]
        Connectivity of neighborhood for each superpixel. Default 8.
    
    Outputs
    -------
    Am: np.ndarray (int) shape:(K x K)
        Adjacency matrix containing values indicating adjacent superpixels
        in theneighborhood. 
    Al: list len(K)
        List containing adjacent indices for each superpixel neighborhood.
    """
    rows, cols = labels.shape[0], labels.shape[1]

    r = []
    c = []
    val = []

    if connectivity == 8:

        for i in range(0, rows - 1):
            r.append(labels[i, 0])
            c.append(labels[i, 1])
            val.append(1)

            r.append(labels[i, 0])
            c.append(labels[i + 1, 0])
            val.append(1)

            r.append(labels[i, 0])
            c.append(labels[i + 1, 1])
            val.append(1)

            for j in range(1, cols - 1):

                r.append(labels[i, j])
                c.append(labels[i, j + 1])
                val.append(1)

                r.append(labels[i, j])
                c.append(labels[i + 1, j - 1])
                val.append(1)

                r.append(labels[i, j])
                c.append(labels[i + 1, j])
                val.append(1)

                r.append(labels[i, j])
                c.append(labels[i + 1, j + 1])
                val.append(1)

    elif connectivity == 4:

        for i in range(0, rows - 1):
            for j in range(1, cols - 1):
                r.append(labels[i, j])
                c.append(labels[i, j + 1])
                val.append(1)

                r.append(labels[i, j])
                c.append(labels[i + 1, j])
                val.append(1)

    else:
        print("Error: Connectivity value other than 4 or 8")

    Am = (
        sp.sparse.csr_matrix((val, (r, c)), shape=(K, K))
        .astype(bool)
        .astype(int)
        .toarray()
    )

    for i in range(0, K):
        Am[i, i] = 0

    Am = Am | np.transpose(Am)

    Al = []

    for i in range(0, K):
        Al.append(np.concatenate(np.argwhere(Am[i,])))

    return Am, Al


def ComputeCenters(
    img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_], K: int
) -> npt.NDArray[np.float64]:
    """
    Computes superpixel centers.

    Inputs
    ------
    img: np.ndarray (float) shape:(n x D)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(n x 1)?
        NumPy array containing superpixel label of each pixel.
    K: int
        Number of superpixels within the image.

    Outputs
    -------
    centers: np.ndarray (float) shape:(K x D)
        Superpixel centers in feature space.
    """

    centers = np.zeros([K, img.shape[1]])

    for k in range(0, K):
        clust = img[labels == k]
        centers[k, 0 : img.shape[1]] = np.mean(clust, 0)

    return centers
