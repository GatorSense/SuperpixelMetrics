import numpy as np
import numpy.typing as npt
from typing import List
from scipy.spatial.distance import pdist
import utils


def I(img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]) -> float:
    """
    Compute I index.

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    I index score.
    """
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    C = np.mean(img, 0)
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    centers = utils.ComputeCenters(img, labels)
    K = utils.get_K(labels)

    imageDISP = np.sum(np.linalg.norm((img - C), None, 1))
    dwtn = 0.00

    for k in range(0, K):
        clust = img[labels == k]

        dwtnk = np.sum(np.linalg.norm(clust - centers[k], None, 1))
        dwtn = dwtn + dwtnk

    score = ((np.max(pdist(centers)) * imageDISP) / (dwtn * K)) ** 2

    return score


def LocalI(
    img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]
) -> npt.NDArray[np.float64]:
    """
    Compute Local I index.
    # Include citation

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    kscores: np.ndarray (float) shape:(K x 1)
        Local I index score for each superpixel.
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
