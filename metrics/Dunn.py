import numpy as np
from scipy.spatial.distance import pdist
import numpy.typing as npt
from typing import List


def Dunn(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    centers: npt.NDArray[np.float64],
) -> float:
    """
    Compute Dunn index.
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
        Dunn index score.
    """

    dunn = np.full([K, 1], 0.00)
    diamMAX = 0.00

    for k in range(0, K):
        clust = img[labels == k]

        if np.max(pdist(clust)) > diamMAX:
            diamMAX = np.max(pdist(clust))

    for k in range(0, K):

        dunnMIN = float("inf")
        for j in range(0, K):

            if j == k:
                continue

            dn = np.linalg.norm((centers[k] - centers[j]))
            if (dn / diamMAX) < dunnMIN:
                dunnMIN = dn / diamMAX
                dunn[k] = dunnMIN

    score = np.min(dunn)

    return score


def LocalDunn(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    centers: npt.NDArray[np.float64],
    Al: List[int],
) -> npt.NDArray[np.float64]:
    """
    Compute Local Dunn index.
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
        Local Dunn score for each superpixel.
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clustk = img[labels == k]
        nhbrs = Al[k]
        dunnMIN = float("inf")
        diamMAX = np.max(pdist(clustk))

        for j in nhbrs:
            clustj = img[labels == j]

            if np.max(pdist(clustj)) > diamMAX:
                diamMAX = np.max(pdist(clustj))

        for j in nhbrs:
            dn = np.linalg.norm((centers[k] - centers[j]))

            if (dn / diamMAX) < dunnMIN:
                dunnMIN = dn / diamMAX

        kscores[k] = dunnMIN

    return kscores
