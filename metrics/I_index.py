import numpy as np
import numpy.typing as npt
from typing import List
from scipy.spatial.distance import pdist


def I(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    C: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
) -> float:
    """
    Compute I index.

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
    I index score.
    """

    imageDISP = np.sum(np.linalg.norm((img - C), None, 1))
    dwtn = 0.00

    for k in range(0, K):
        clust = img[labels == k]

        dwtnk = np.sum(np.linalg.norm(clust - centers[k], None, 1))
        dwtn = dwtn + dwtnk

    score = ((np.max(pdist(centers)) * imageDISP) / (dwtn * K)) ** 2

    return score


def LocalI(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    centers: npt.NDArray[np.float64],
    Al: List[int],
) -> npt.NDArray[np.float64]:
    """
    Compute Local I index.
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
        Local I index score for each superpixel.
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        nhbrs = np.append(Al[k], k)
        nhbrhd = img[np.in1d(labels, nhbrs)]
        Cnk = np.mean(nhbrhd, 0)

        nhbrhdDISP = np.sum(np.linalg.norm((nhbrhd - Cnk), None, 1))

        dwtn = 0.00

        for j in nhbrs:
            clust = img[labels == j]

            dwtnj = np.sum(np.linalg.norm(clust - centers[j], None, 1))
            dwtn = dwtn + dwtnj

        kscores[k] = (
            (np.max(pdist(centers[nhbrs])) * nhbrhdDISP) / (dwtn * len(nhbrs))
        ) ** 2

    return kscores
