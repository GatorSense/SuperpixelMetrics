import numpy as np
import numpy.typing as npt
from typing import List
from scipy.spatial.distance import pdist
import utils


def XieBeni(img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]) -> float:
    """
    Compute Xie Beni index.
    #Include citation

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    Xie Beni score.
    """
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    centers = utils.ComputeCenters(img, labels)
    K = utils.get_K(labels)
    n = utils.get_n(img)

    dwtn = 0
    dn = pdist(centers) ** 2

    for k in range(0, K):
        clust = img[labels == k]
        dwtn = dwtn + np.sum(np.linalg.norm((clust - centers[k]), None, 1) ** 2)

    score = dwtn / (n * (np.min(dn)))

    return score


def LocalXieBeni(
    img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]
) -> npt.NDArray[np.float64]:
    """
    Compute local Xie Beni index for each superpixel.
    #Include citation

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    kscores: np.ndarray (float) shape:(K x 1)
        Local Xie Beni index for each superpixel.
    """
    _, Al = utils.ComputeAdjacency(labels)
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    centers = utils.ComputeCenters(img, labels)
    K = utils.get_K(labels)

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
