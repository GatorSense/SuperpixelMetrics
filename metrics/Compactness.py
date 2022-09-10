import numpy as np
import numpy.typing as npt
from skimage.measure import regionprops
from metrics.utils import *


def Compactness(img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]) -> float:
    """
    Compute superpixel compactness.
    #Include reference.

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    Superpixel compactness score.
    """
    properties = regionprops(labels+1)

    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    K = get_K(labels)
    n = get_n(img)

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clust = img[np.where(labels == k)[0], :]
        nk = clust.shape[0]
        pk = properties[k]["perimeter"]

        kscores[k] = ((nk**2) * 4 * np.pi) / (pk**2)

    score = np.sum(kscores) / n

    return score
