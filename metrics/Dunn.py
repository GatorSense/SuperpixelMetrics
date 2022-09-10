import numpy as np
from scipy.spatial.distance import pdist
import numpy.typing as npt
from metrics.utils import *


def Dunn(img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]) -> float:
    """
    Compute Dunn index.
    # Include citation

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    score: float
        Dunn index score.
    """
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    centers = ComputeCenters(img, labels)
    K = get_K(labels)

    dunn = np.full([K, 1], 0.00)
    diamMAX = 0.00

    for k in range(0, K):
        clust = img[np.where(labels == k)[0], :]

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
    img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]
) -> npt.NDArray[np.float64]:
    """
    Compute Local Dunn index.
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
        Local Dunn score for each superpixel.
    """
    _, Al = ComputeAdjacency(labels)
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    centers = ComputeCenters(img, labels)
    K = get_K(labels)

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clustk = img[np.where(labels == k)[0], :]
        nhbrs = Al[k]
        dunnMIN = float("inf")
        diamMAX = np.max(pdist(clustk))

        for j in nhbrs:
            clustj = img[np.where(labels == j)[0], :]

            if np.max(pdist(clustj)) > diamMAX:
                diamMAX = np.max(pdist(clustj))

        for j in nhbrs:
            dn = np.linalg.norm((centers[k] - centers[j]))

            if (dn / diamMAX) < dunnMIN:
                dunnMIN = dn / diamMAX

        kscores[k] = dunnMIN

    return kscores
