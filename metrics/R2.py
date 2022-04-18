import numpy as np
import numpy.typing as npt
from typing import List
import utils


def RSquared(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_]
) -> float:
    """
    Compute R^2

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    R^2
    """
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    centers = utils.ComputeCenters(img, labels)
    K = utils.get_K(labels)

    num = 0

    for k in range(0, K):
        clust = img[labels == k]
        nk = clust.shape[0]

        num = num + nk * np.linalg.norm(centers[k] - C) ** 2

    score = num / (np.sum(np.linalg.norm((img - C), None, 1) ** 2))

    return score


def LocalRSquared(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_]
) -> npt.NDArray[np.float64]:
    """
    Compute local R^2.

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    kscores: np.ndarray (float) shape:(K x 1)
        Local R^2 value for each superpixel.
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

        num = 0
        for j in nhbrs:
            clustj = img[labels == j]
            nj = clustj.shape[0]

            num = num + (nj * np.linalg.norm(centers[j] - Cnk) ** 2)

        kscores[k] = num / (np.sum(np.linalg.norm((nhbrhd - Cnk), None, 1) ** 2))

    return kscores
