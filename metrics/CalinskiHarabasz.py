import numpy as np
import numpy.typing as npt
import utils


def CalinskiHarabasz(
    img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]
) -> float:
    """
    Compute Calinski Harabasz index.
    # Include citation

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    Calinski Harabasz score.
    """
    img = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
    C = np.mean(img, 0)
    centers = utils.ComputeCenters(img, labels)
    K = utils.get_K(labels)
    n = utils.get_n(img)

    ssbtwn = 0
    sswtn = 0

    for k in range(0, K):
        clust = img[labels == k]
        nk = clust.shape[0]
        ck = centers[k]

        ssbtwn = ssbtwn + nk * np.linalg.norm((ck - C)) ** 2
        sswtn = sswtn + np.sum(np.linalg.norm((clust - ck), None, 1) ** 2)

    score = (ssbtwn / (K - 1)) / (sswtn / (n - K))

    return score


def LocalCalinskiHarabasz(
    img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]
) -> npt.NDArray[np.float64]:
    """
    Compute local Calinski Harabasz index for each superpixel.

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    kscores: np.ndarray (float) shape:(K x 1)
        Local Calinski Harabasz score for each superpixel.
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
        nnk = nhbrhd.shape[0]
        Cnk = np.mean(nhbrhd, 0)

        ssbtwn = 0
        sswtn = 0

        for m in nhbrs:
            clust = img[labels == m]
            nm = clust.shape[0]
            cm = centers[m]

            ssbtwn = ssbtwn + nm * np.linalg.norm((cm - Cnk)) ** 2
            sswtn = sswtn + np.sum(np.linalg.norm((clust - cm), None, 1) ** 2)

        kscores[k] = (ssbtwn / (len(nhbrs) - 1)) / (sswtn / (nnk - len(nhbrs)))

    return kscores


def LocalCalinskiHarabasz2(
    img: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]
) -> npt.NDArray[np.float64]:
    """
    Compute variant of Local Calinski Harabasz index for each superpixel.

    Inputs
    ------
    img: np.ndarray (float) shape:(R x C x d)
        Reshaped NumPy array of image data with each row containing a pixel and its features.
    labels: np.ndarray (int) shape:(R x C)
        NumPy array containing superpixel label of each pixel.

    Output
    ------
    kscores: np.ndarray (float) shape:(K x 1)
        Variant of Local Calinski Harabasz score for each superpixel.
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
        nnk = nhbrhd.shape[0]
        Cnk = np.mean(nhbrhd, 0)

        ssbtwn = 0
        sswtn = 0

        for m in nhbrs:
            clust = img[labels == m]
            nm = clust.shape[0]
            cm = centers[m]

            ssbtwn = ssbtwn + nm * np.linalg.norm((cm - Cnk)) ** 2
            sswtn = sswtn + np.sum(np.linalg.norm((clust - cm), None, 1) ** 2)

        kscores[k] = (ssbtwn / (len(nhbrs) - 1)) / (sswtn / (nnk - (nnk / len(nhbrs))))

    return kscores
