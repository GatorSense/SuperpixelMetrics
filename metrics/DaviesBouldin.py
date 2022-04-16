import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist
from typing import List


def DaviesBouldin(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    centers: npt.NDArray[np.float64],
) -> float:
    """
    Compute Davies Bouldin index.
    # Include citation

    Inputs
    ------
    img: np.ndarray (float) shape:(n x D)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(n x 1)?
        NumPy array containing superpixel label of each pixel.
    K: int
        Number of superpixels within the image.
    centers: np.ndarray (float) shape:(K x D)
        Superpixel centers in feature space. 
    
    Output
    ------
    score: float
        Davies Bouldin score.
    """

    db = np.full([K, 1], -1.00)

    for k in range(0, K):
        clustk = img[labels == k]
        nk = clustk.shape[0]
        ck = centers[k]
        dwtnk = np.sum(np.linalg.norm((clustk - ck), None, 1)) / nk

        for j in range(0, K):

            if j == k:
                continue

            clustj = img[labels == j]
            nj = clustj.shape[0]
            cj = centers[j]
            dwtnj = np.sum(np.linalg.norm((clustj - cj), None, 1)) / nj
            dn = np.linalg.norm((ck - cj))

            dbj = (dwtnk + dwtnj) / dn

            if dbj > db[k]:
                db[k] = dbj

    score = np.mean(db)

    return score


def LocalDaviesBouldin(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    centers: npt.NDArray[np.float64],
    Al: List[int],
):
    """
    Compute Local Davies Bouldin index.
    # Include citation

    Inputs
    ------
    img: np.ndarray (float) shape:(n x D)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(n x 1)?
        NumPy array containing superpixel label of each pixel.
    K: int
        Number of superpixels within the image.
    centers: np.ndarray (float) shape:(K x D)
        Superpixel centers in feature space.
    Al: list len(K)
        List containing adjacent indices for each superpixel neighborhood.
    
    Output
    ------
    kscores: np.ndarray (float) shape:(K x 1)
        Local Davies Bouldin score for each superpixel.
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clustk = img[labels == k]
        ck = centers[k]

        nhbrs = Al[k]

        diamk = np.max(pdist(clustk))
        dbMAX = 0

        for j in nhbrs:
            clustj = img[labels == j]
            cj = centers[j]

            if j == k:
                continue

            dn = np.linalg.norm((ck - cj))
            diamj = np.max(pdist(clustj))

            if ((diamk + diamj) / dn) > dbMAX:
                dbMAX = (diamk + diamj) / dn

        kscores[k] = dbMAX

    return kscores
