import numpy as np
import numpy.typing as npt
from typing import List


def RSquared(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    C: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
) -> float:
    """
    Compute R^2

    Inputs
    ------
    img: np.ndarray (float) shape:(n x D)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(n x 1)?
        NumPy array containing superpixel label of each pixel. 
    K: int
        Number of superpixels within the image.
    C: np.ndarray (float) shape: (1 x D)
        Mean vector of image.
    centers: np.ndarray (float) shape:(K x D)
        Superpixel centers in feature space.       
    
    Output
    ------
    R^2
    """

    num = 0

    for k in range(0, K):
        clust = img[labels == k]
        nk = clust.shape[0]

        num = num + nk * np.linalg.norm(centers[k] - C) ** 2

    score = num / (np.sum(np.linalg.norm((img - C), None, 1) ** 2))

    return score


def LocalRSquared(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    centers: npt.NDArray[np.float64],
    Al: List[int],
) -> npt.NDArray[np.float64]:
    """
    Compute local R^2.

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
        Local R^2 value for each superpixel.
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        nhbrs = np.append(Al[k], k)
        nhbrhd = img[np.in1d(labels, nhbrs)]
        Cnk = np.mean(nhbrhd, 0)

        num = 0
        for j in nhbrs:
            clustj = img[labels == j]
            nj = clustj.shape[0]

            num = num + (nj * np.linalg.norm(centers[j] - Cnk) ** 2)

        kscores[k] = num / (np.sum(np.linalg.norm((nhbrhd - Cnk), None, 1) ** 2))

    return kscores
