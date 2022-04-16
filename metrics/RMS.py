import numpy as np
import numpy.typing as npt
from typing import List


def RootMeanSquaredStandardDeviation(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    centers: npt.NDArray[np.float64],
) -> float:
    """
    Compute Root Mean Squared Standard Deviation.

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
    Root Mean Squared Standard Deviation score.
    """

    dwtn = 0
    denom = 0

    for k in range(0, K):
        clust = img[labels == k]
        nk = clust.shape[0]
        dwtn = dwtn + np.sum(np.linalg.norm((clust - centers[k]), None, 1) ** 2)
        denom = denom + (nk - 1)

    score = np.sqrt(dwtn / denom)

    return score


def LocalRootMeanSquaredStandardDeviation(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    centers: npt.NDArray[np.float64],
    Al: List[int],
) -> npt.NDArray[np.float64]:
    """
    Compute Local Root Mean Squared Standard Deviation.

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
        Local RMSSTD value for each superpixel.
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        nhbrs = np.append(Al[k], k)
        dwtn = 0
        denom = 0

        for j in nhbrs:
            clustj = img[labels == j]
            nj = clustj.shape[0]

            dwtn = dwtn + np.sum(np.linalg.norm((clustj - centers[j]), None, 1) ** 2)
            denom = denom + (nj - 1)

        kscores[k] = np.sqrt(dwtn / denom)

    return kscores