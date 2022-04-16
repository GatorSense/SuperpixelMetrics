import numpy as np
import numpy.typing as npt


def Variance(
    img: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    K: int,
    centers: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute Superpixel variance.

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
    
    Output
    ------
    kscores: np.ndarray (float) shape:(K x 1)
        Local variance value for each superpixel.
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clust = img[labels == k]
        nk = clust.shape[0]

        num = np.sum(np.linalg.norm((clust - centers[k]), None, 1) ** 2)

        kscores[k] = num / nk

    return kscores
