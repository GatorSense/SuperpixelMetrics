import numpy as np
import numpy.typing as npt
from typing import List
from metrics.utils import *


def RMSSTD(img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]) -> float:
    """
    Compute Root Mean Squared Standard Deviation.

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    Root Mean Squared Standard Deviation score.
    """
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    centers = ComputeCenters(img, labels)
    K = get_K(labels)

    dwtn = 0
    denom = 0

    for k in range(0, K):
        clust = img[np.where(labels == k)[0], :]
        nk = clust.shape[0]
        dwtn = dwtn + np.sum(np.linalg.norm((clust - centers[k]), None, 1) ** 2)
        denom = denom + (nk - 1)

    score = np.sqrt(dwtn / denom)

    return score


def LocalRMSSTD(
    img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]
) -> npt.NDArray[np.float64]:
    """
    Compute Local Root Mean Squared Standard Deviation.

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    kscores: np.ndarray (float) shape:(K x 1)
        Local RMSSTD value for each superpixel.
    """
    _, Al = ComputeAdjacency(labels)
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    centers = ComputeCenters(img, labels)
    K = get_K(labels)

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        nhbrs = np.append(Al[k], k)
        dwtn = 0
        denom = 0

        for j in nhbrs:
            clustj = img[np.where(labels == j)[0], :]
            nj = clustj.shape[0]

            dwtn = dwtn + np.sum(np.linalg.norm((clustj - centers[j]), None, 1) ** 2)
            denom = denom + (nj - 1)

        kscores[k] = np.sqrt(dwtn / denom)

    return kscores
