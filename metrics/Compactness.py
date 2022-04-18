import numpy as np
import numpy.typing as npt
from skimage.measure import regionprops
import utils


def Compactness(
    img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]
) -> float:
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
    properties = regionprops(labels)  

    img = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0]*labels.shape[1],1))
    K = utils.get_K(labels)
    n = utils.get_n(img)
    
    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clust = img[labels == k]
        nk = clust.shape[0]
        pk = properties[k]["perimeter"]

        kscores[k] = ((nk ** 2) * 4 * np.pi) / (pk ** 2)

    score = np.sum(kscores) / n

    return score
