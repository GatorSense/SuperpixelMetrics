
# Comparison of Quantitative Evaluation Metrics for Superpixel Segmentation 

In this repository, we provide implementations for 17 quantative evaluation metrics for superpixel segmentation as well as code to compute superpixel centers, an adjacency list, an adjacency matrix, and superpixel properties.

The required packages can be found in requirements.txt. To install those, you can use the command:

```pip install -r requirements.txt```

***

## I. Berkeley Segmentation Dataset 500

Metric validation experiments were conducted on the 200 images in the BSDS500 training set. To access these images, please proceed to the [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) website.

## II. Functions

```python 
def ComputeAdjacency(labels, K, connectivity) 
```
Inputs: 
- K = number of superpixels
- labels = RxC where R is the number of rows in an image and C is the number #of columns
- connectivity = 4 or 8

Outputs:
- Am = adjacency matrix, RxC where R is the number of rows in an image and C is the number of columns
- Al = adjacency list of length K

```python
 def ComputeProperties(labels)
 ```
Inputs:     
- labels = RxC where R is the number of rows in an image and C is the number of columns (values must begin at one)

Output: 
- refer to [scikit image](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)

```python
def ComputeCenters(img, labels, K)
 ```
Inputs: 
- K = number of superpixels
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image

Output:
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel

```python
def CalinskiHarabasz(img, labels, n, K, C, centers) 
```
Inputs: 
- n = number of pixels in image
- K = number of superpixels
- C = mean vector of image 1xD where D is the dimensionality of a pixel
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel

Output: 
- score = metric score

```python
def LocalCalinskiHarabasz(img, labels, K, centers, Al)
```
Inputs: 
- K = number of superpixels
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
- Al = adjacency list of length K

Output: 
- kscores = metric scores for each superpixel, length K

```python
 def LocalCalinskiHarabasz2(img, labels, K, centers, Al)
```
Inputs: 
- K = number of superpixels
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
- Al = adjacency list of length K

Output: 
- kscores = metric scores for each superpixel, length K

```python
def DaviesBouldin(img, labels, K, centers)
```
Inputs: 
- K = number of superpixels
- C = mean vector of image 1xD where D is the dimensionality of a pixel
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel

Output: 
- score = metric score

```python
def LocalDaviesBouldin(img, labels, K, centers, Al)
```
Inputs: 
- K = number of superpixels
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
- Al = adjacency list of length K

Output: 
- kscores = metric scores for each superpixel, length K

```python
def Dunn(img, labels, K, centers)
```
Inputs: 
- K = number of superpixels
- C = mean vector of image 1xD where D is the dimensionality of a pixel
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel

Output: 
- score = metric score

```python
def LocalDunn(img, labels, K, centers, Al)
```
Inputs: 
- K = number of superpixels
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
- Al = adjacency list of length K

Output: 
- kscores = metric scores for each superpixel, length K

```python
def I(img, labels, K, C, centers)
```
Inputs: 
- K = number of superpixels
- C = mean vector of image 1xD where D is the dimensionality of a pixel
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel

Output: 
- score = metric score

```python 
def LocalI(img, labels, K, centers, Al)
```
Inputs: 
- K = number of superpixels
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
- Al = adjacency list of length K

Output: 
- kscores = metric scores for each superpixel, length K

```python
def RSquared(img, labels, K, C, centers)
```
Inputs: 
- K = number of superpixels
- C = mean vector of image 1xD where D is the dimensionality of a pixel
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel

Output: 
- score = metric score

```python
def LocalRSquared(img, labels, K, centers, Al)
```
Inputs: 
- K = number of superpixels
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
- Al = adjacency list of length K

Output: 
- kscores = metric scores for each superpixel, length K

```python
def RootMeanSquaredStandardDeviation(img, labels, K, centers)
```
Inputs: 
- K = number of superpixels
- C = mean vector of image 1xD where D is the dimensionality of a pixel
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel

Output: 
- score = metric score

```python
def LocalRootMeanSquaredStandardDeviation(img, labels, K, centers, Al)
```
Inputs: 
- K = number of superpixels
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
- Al = adjacency list of length K

Output: 
- kscores = metric scores for each superpixel, length K

```python
def XieBeni(img, labels, n, K, centers)
```
Inputs: 
- n = number of pixels in image
- K = number of superpixels
- C = mean vector of image 1xD where D is the dimensionality of a pixel
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel

Output: 
- score = metric score

```python
def LocalXieBeni(img, labels, K, centers, Al)
```
Inputs: 
- K = number of superpixels
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
- Al = adjacency list of length K

Output: 
- kscores = metric scores for each superpixel, length K

```python
def Variance(img, labels, K, centers)
```
Inputs: 
- K = number of superpixels
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- centers = KxD where K is the number of superpixels and D is the dimensionality of a pixel
- Al = adjacency list of length K

Output: 
- kscores = metric scores for each superpixel, length K

```python
def Compactness(img, labels, n, K, properties)
```
Inputs: 
- n = number of pixels in image
- K = number of superpixels
- img = nxD where n is the number of pixels in the image and D is the dimensionality of a pixel
- labels = nx1 where n is the number of pixels in the image
- properties = refer to [scikit image](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)

Output: 
- score = metric score

***

## III. Citing

If you use this code, please cite ..

This code uses the function regionprops from [scikit image](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).