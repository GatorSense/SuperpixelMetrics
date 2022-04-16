import numpy as np
import numpy.typing as npt
from skimage.measure import regionprops


def Compactness(
    img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_], n: float, K: float
) -> float:
    """
    Compute superpixel compactness.
    #Include reference.

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
    
    Output
    ------
    Superpixel compactness score.
    """
    properties = regionprops(labels)
    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clust = img[labels == k]
        nk = clust.shape[0]
        pk = properties[k]["perimeter"]

        kscores[k] = ((nk ** 2) * 4 * np.pi) / (pk ** 2)

    score = np.sum(kscores) / n

    return score
