import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist
from metrics.utils import *


def DaviesBouldin(img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]) -> float:
    """
    Compute Davies Bouldin index.
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
        Davies Bouldin score.
    """
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    centers = ComputeCenters(img, labels)
    K = get_K(labels)

    db = np.full([K, 1], -1.00)

    for k in range(0, K):
        clustk = img[np.where(labels == k)[0], :]
        nk = clustk.shape[0]
        ck = centers[k]
        dwtnk = np.sum(np.linalg.norm((clustk - ck), None, 1)) / nk

        for j in range(0, K):

            if j == k:
                continue

            clustj = img[np.where(labels == j)[0],:]
            nj = clustj.shape[0]
            cj = centers[j]
            dwtnj = np.sum(np.linalg.norm((clustj - cj), None, 1)) / nj
            dn = np.linalg.norm((ck - cj))

            dbj = (dwtnk + dwtnj) / dn

            if dbj > db[k]:
                db[k] = dbj

    score = np.mean(db)

    return score


def LocalDaviesBouldin(img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]):
    """
    Compute Local Davies Bouldin index.
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
        Local Davies Bouldin score for each superpixel.
    """
    _, Al = ComputeAdjacency(labels)
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    centers = ComputeCenters(img, labels)
    K = get_K(labels)

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clustk = img[np.where(labels == k)[0], :]
        ck = centers[k]

        nhbrs = Al[k]

        diamk = np.max(pdist(clustk))
        dbMAX = 0

        for j in nhbrs:
            clustj = img[np.where(labels == j)[0], :]
            cj = centers[j]

            if j == k:
                continue

            dn = np.linalg.norm((ck - cj))
            diamj = np.max(pdist(clustj))

            if ((diamk + diamj) / dn) > dbMAX:
                dbMAX = (diamk + diamj) / dn

        kscores[k] = dbMAX

    return kscores
