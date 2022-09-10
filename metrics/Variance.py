import numpy as np
import numpy.typing as npt
from metrics.utils import *


def Variance(
    img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]
) -> npt.NDArray[np.float64]:
    """
    Compute Superpixel variance.

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    kscores: np.ndarray (float) shape:(K x 1)
        Local variance value for each superpixel.
    """
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    centers = ComputeCenters(img, labels)
    K = get_K(labels)

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clust = img[np.where(labels == k)[0], :]
        nk = clust.shape[0]

        num = np.sum(np.linalg.norm((clust - centers[k]), None, 1) ** 2)

        kscores[k] = num / nk

    return kscores
