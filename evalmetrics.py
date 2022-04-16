import scipy as sp
import numpy as np
from scipy.spatial.distance import pdist
from skimage.measure import regionprops


def ComputeAdjacency(
    labels: np.ndarray, K: int, connectivity=8
) -> tuple(np.ndarray, list):
    """
    Compute adjacency matrix

    Inputs
    ------
    labels: np.ndarray (int) size(image)
        NumPy array containing superpixel label of each pixel.
    K: int
        Number of superpixels within the image.
    connectivity: int [4 | 8]
        Connectivity of neighborhood for each superpixel. Default 8.
    
    Outputs
    -------
    Am: np.ndarray (int) size(K x K)
        Adjacency matrix containing values indicating adjacent superpixels
        in theneighborhood. 
    Al: list len(K)
        List containing adjacent indices for each superpixel neighborhood.
    
    """
    rows, cols = labels.shape[0], labels.shape[1]

    r = []
    c = []
    val = []

    if connectivity == 8:

        for i in range(0, rows - 1):
            r.append(labels[i, 0])
            c.append(labels[i, 1])
            val.append(1)

            r.append(labels[i, 0])
            c.append(labels[i + 1, 0])
            val.append(1)

            r.append(labels[i, 0])
            c.append(labels[i + 1, 1])
            val.append(1)

            for j in range(1, cols - 1):

                r.append(labels[i, j])
                c.append(labels[i, j + 1])
                val.append(1)

                r.append(labels[i, j])
                c.append(labels[i + 1, j - 1])
                val.append(1)

                r.append(labels[i, j])
                c.append(labels[i + 1, j])
                val.append(1)

                r.append(labels[i, j])
                c.append(labels[i + 1, j + 1])
                val.append(1)

    elif connectivity == 4:

        for i in range(0, rows - 1):
            for j in range(1, cols - 1):
                r.append(labels[i, j])
                c.append(labels[i, j + 1])
                val.append(1)

                r.append(labels[i, j])
                c.append(labels[i + 1, j])
                val.append(1)

    else:
        print("Error: Connectivity value other than 4 or 8")

    Am = (
        sp.sparse.csr_matrix((val, (r, c)), shape=(K, K))
        .astype(bool)
        .astype(int)
        .toarray()
    )

    for i in range(0, K):
        Am[i, i] = 0

    Am = Am | np.transpose(Am)

    Al = []

    for i in range(0, K):
        Al.append(np.concatenate(np.argwhere(Am[i,])))

    return Am, Al


def ComputeProperties(labels):
    """
    labels = RxC where R is the number of rows in an image and C is the number of columns (values must begin at one)

    returns superpixel properties, refer to https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    """

    properties = regionprops(labels)

    return properties


def ComputeCenters(img, labels, K):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image

    returns centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    centers = np.zeros([K, img.shape[1]])

    for k in range(0, K):
        clust = img[labels == k]
        centers[k, 0 : img.shape[1]] = np.mean(clust, 0)

    return centers


def CalinskiHarabasz(img, labels, n, K, C, centers):
    """
    n = number of pixels in image
    K = number of superpixels
    C = mean vector of image 1xD where D is the dimensionality of a pixel
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel 
    """

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


def LocalCalinskiHarabasz(img, labels, K, centers, Al):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    Al = adjacency list of length K
    """

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


def LocalCalinskiHarabasz2(img, labels, K, centers, Al):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel 
    """

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


def DaviesBouldin(img, labels, K, centers):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    db = np.full([K, 1], -1.00)

    for k in range(0, K):
        clustk = img[labels == k]
        nk = clustk.shape[0]
        ck = centers[k]
        dwtnk = np.sum(np.linalg.norm((clustk - ck), None, 1)) / nk

        for j in range(0, K):

            if j == k:
                continue

            clustj = img[labels == j]
            nj = clustj.shape[0]
            cj = centers[j]
            dwtnj = np.sum(np.linalg.norm((clustj - cj), None, 1)) / nj
            dn = np.linalg.norm((ck - cj))

            dbj = (dwtnk + dwtnj) / dn

            if dbj > db[k]:
                db[k] = dbj

    score = np.mean(db)

    return score


def LocalDaviesBouldin(img, labels, K, centers, Al):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clustk = img[labels == k]
        ck = centers[k]

        nhbrs = Al[k]

        diamk = np.max(pdist(clustk))
        dbMAX = 0

        for j in nhbrs:
            clustj = img[labels == j]
            cj = centers[j]

            if j == k:
                continue

            dn = np.linalg.norm((ck - cj))
            diamj = np.max(pdist(clustj))

            if ((diamk + diamj) / dn) > dbMAX:
                dbMAX = (diamk + diamj) / dn

        kscores[k] = dbMAX

    return kscores


def Dunn(img, labels, K, centers):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    dunn = np.full([K, 1], 0.00)
    diamMAX = 0.00

    for k in range(0, K):
        clust = img[labels == k]

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


def LocalDunn(img, labels, K, centers, Al):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clustk = img[labels == k]
        nhbrs = Al[k]
        dunnMIN = float("inf")
        diamMAX = np.max(pdist(clustk))

        for j in nhbrs:
            clustj = img[labels == j]

            if np.max(pdist(clustj)) > diamMAX:
                diamMAX = np.max(pdist(clustj))

        for j in nhbrs:
            dn = np.linalg.norm((centers[k] - centers[j]))

            if (dn / diamMAX) < dunnMIN:
                dunnMIN = dn / diamMAX

        kscores[k] = dunnMIN

    return kscores


def I(img, labels, K, C, centers):
    """
    K = number of superpixels
    C = mean vector of image 1xD where D is the dimensionality of a pixel
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    imageDISP = np.sum(np.linalg.norm((img - C), None, 1))
    dwtn = 0.00

    for k in range(0, K):
        clust = img[labels == k]

        dwtnk = np.sum(np.linalg.norm(clust - centers[k], None, 1))
        dwtn = dwtn + dwtnk

    score = ((np.max(pdist(centers)) * imageDISP) / (dwtn * K)) ** 2

    return score


def LocalI(img, labels, K, centers, Al):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        nhbrs = np.append(Al[k], k)
        nhbrhd = img[np.in1d(labels, nhbrs)]
        Cnk = np.mean(nhbrhd, 0)

        nhbrhdDISP = np.sum(np.linalg.norm((nhbrhd - Cnk), None, 1))

        dwtn = 0.00

        for j in nhbrs:
            clust = img[labels == j]

            dwtnj = np.sum(np.linalg.norm(clust - centers[j], None, 1))
            dwtn = dwtn + dwtnj

        kscores[k] = (
            (np.max(pdist(centers[nhbrs])) * nhbrhdDISP) / (dwtn * len(nhbrs))
        ) ** 2

    return kscores


def RSquared(img, labels, K, C, centers):
    """
    K = number of superpixels
    C = mean vector of image 1xD where D is the dimensionality of a pixel
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    num = 0

    for k in range(0, K):
        clust = img[labels == k]
        nk = clust.shape[0]

        num = num + nk * np.linalg.norm(centers[k] - C) ** 2

    score = num / (np.sum(np.linalg.norm((img - C), None, 1) ** 2))

    return score


def LocalRSquared(img, labels, K, centers, Al):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

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


def RootMeanSquaredStandardDeviation(img, labels, K, centers):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    dwtn = 0
    denom = 0

    for k in range(0, K):
        clust = img[labels == k]
        nk = clust.shape[0]
        dwtn = dwtn + np.sum(np.linalg.norm((clust - centers[k]), None, 1) ** 2)
        denom = denom + (nk - 1)

    score = np.sqrt(dwtn / denom)

    return score


def LocalRootMeanSquaredStandardDeviation(img, labels, K, centers, Al):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        nhbrs = np.append(Al[k], k)
        dwtn = 0
        denom = 0

        for j in nhbrs:
            clustj = img[labels == j]
            nj = clustj.shape[0]

            dwtn = dwtn + np.sum(np.linalg.norm((clustj - centers[j]), None, 1) ** 2)
            denom = denom + (nj - 1)

        kscores[k] = np.sqrt(dwtn / denom)

    return kscores


def XieBeni(img, labels, n, K, centers):
    """
    n = number of pixels in image
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    dwtn = 0
    dn = pdist(centers) ** 2

    for k in range(0, K):
        clust = img[labels == k]
        dwtn = dwtn + np.sum(np.linalg.norm((clust - centers[k]), None, 1) ** 2)

    score = dwtn / (n * (np.min(dn)))

    return score


def LocalXieBeni(img, labels, K, centers, Al):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        nhbrs = np.append(Al[k], k)
        nhbrhd = img[np.in1d(labels, nhbrs)]
        nnk = nhbrhd.shape[0]

        dwtn = 0
        dn = pdist(centers[nhbrs]) ** 2

        for j in nhbrs:
            clustj = img[labels == j]
            dwtn = dwtn + np.sum(np.linalg.norm((clustj - centers[j]), None, 1) ** 2)

        kscores[k] = dwtn / (nnk * np.min(dn))

    return kscores


def Variance(img, labels, K, centers):
    """
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clust = img[labels == k]
        nk = clust.shape[0]

        num = np.sum(np.linalg.norm((clust - centers[k]), None, 1) ** 2)

        kscores[k] = num / nk

    return kscores


def Compactness(img, labels, n, K, properties):
    """
    n = number of pixels in image
    K = number of superpixels
    img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
    labels = nx1 where n is the number of pixels in the image
    """

    kscores = np.full([K, 1], 0.00)

    for k in range(0, K):
        clust = img[labels == k]
        nk = clust.shape[0]
        pk = properties[k]["perimeter"]

        kscores[k] = ((nk ** 2) * 4 * np.pi) / (pk ** 2)

    score = np.sum(kscores) / n

    return score
