import numpy as np
import numpy.typing as npt
from typing import List


def CalinskiHarabasz(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    n: int,
    K: int,
    C: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
) -> float:
    """
    Compute Calinski Harabasz index.
    # Include citation

    Inputs
    ------
    img: np.ndarray (float) shape:(n x D)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(n x 1)?
        NumPy array containing superpixel label of each pixel. 
    n: int
        Number of pixels in image.
    K: int
        Number of superpixels within the image.
    C: np.ndarray (float) shape: (1 x D)
        Mean vector of image.
    
    Output
    ------
    Calinski Harabasz score.
    """

    ssbtwn = 0
    sswtn = 0

    for k in range(0, K):
        clust = img[labels == k]
        nk = clust.shape[0]
        ck = centers[k]

        ssbtwn = ssbtwn + nk * np.linalg.norm((ck - C)) ** 2
        sswtn = sswtn + np.sum(np.linalg.norm((clust - ck), None, 1) ** 2)

    score = (ssbtwn / (K - 1)) / (sswtn / (n - K))

    return score


def LocalCalinskiHarabasz(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    centers: npt.NDArray[np.float64],
    Al: List[int],
) -> npt.NDArray[np.float64]:
    """
    Compute local Calinski Harabasz index for each superpixel.

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
        Local Calinski Harabasz score for each superpixel.
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):

        nhbrs = np.append(Al[k], k)
        nhbrhd = img[np.in1d(labels, nhbrs)]
        nnk = nhbrhd.shape[0]
        Cnk = np.mean(nhbrhd, 0)

        ssbtwn = 0
        sswtn = 0

        for m in nhbrs:
            clust = img[labels == m]
            nm = clust.shape[0]
            cm = centers[m]

            ssbtwn = ssbtwn + nm * np.linalg.norm((cm - Cnk)) ** 2
            sswtn = sswtn + np.sum(np.linalg.norm((clust - cm), None, 1) ** 2)

        kscores[k] = (ssbtwn / (len(nhbrs) - 1)) / (sswtn / (nnk - len(nhbrs)))

    return kscores


def LocalCalinskiHarabasz2(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    centers: npt.NDArray[np.float64],
    Al: List[int],
) -> npt.NDArray[np.float64]:
    """
    Compute variant of Local Calinski Harabasz index for each superpixel.

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
        Variant of Local Calinski Harabasz score for each superpixel.
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):

        nhbrs = np.append(Al[k], k)
        nhbrhd = img[np.in1d(labels, nhbrs)]
        nnk = nhbrhd.shape[0]
        Cnk = np.mean(nhbrhd, 0)

        ssbtwn = 0
        sswtn = 0

        for m in nhbrs:
            clust = img[labels == m]
            nm = clust.shape[0]
            cm = centers[m]

            ssbtwn = ssbtwn + nm * np.linalg.norm((cm - Cnk)) ** 2
            sswtn = sswtn + np.sum(np.linalg.norm((clust - cm), None, 1) ** 2)

        kscores[k] = (ssbtwn / (len(nhbrs) - 1)) / (sswtn / (nnk - (nnk / len(nhbrs))))

    return kscores
