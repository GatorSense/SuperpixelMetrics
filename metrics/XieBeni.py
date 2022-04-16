import numpy as np
import numpy.typing as npt
from typing import List
from scipy.spatial.distance import pdist


def XieBeni(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    n: int,
    K: int,
    centers: npt.NDArray[np.float64],
) -> float:
    """
    Compute Xie Beni index.
    #Include citation

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
    centers: np.ndarray (float) shape:(K x D)
        Superpixel centers in feature space.   
    
    Output
    ------
    Xie Beni score.
    """

    dwtn = 0
    dn = pdist(centers) ** 2

    for k in range(0, K):
        clust = img[labels == k]
        dwtn = dwtn + np.sum(np.linalg.norm((clust - centers[k]), None, 1) ** 2)

    score = dwtn / (n * (np.min(dn)))

    return score


def LocalXieBeni(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    centers: npt.NDArray[np.float64],
    Al: List[int],
) -> npt.NDArray[np.float64]:
    """
    Compute local Xie Beni index for each superpixel.
    #Include citation

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
        Local Xie Beni index for each superpixel.
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        nhbrs = np.append(Al[k], k)
        nhbrhd = img[np.in1d(labels, nhbrs)]
        nnk = nhbrhd.shape[0]

        dwtn = 0
        dn = pdist(centers[nhbrs]) ** 2

        for j in nhbrs:
            clustj = img[labels == j]
            dwtn = dwtn + np.sum(np.linalg.norm((clustj - centers[j]), None, 1) ** 2)

        kscores[k] = dwtn / (nnk * np.min(dn))

    return kscores
