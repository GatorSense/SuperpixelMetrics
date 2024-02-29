
# Comparison of Quantitative Metrics for Superpixel Segmentation 

In this repository, we provide implementations for 17 quantative evaluation metrics for superpixel segmentation as well as code to compute superpixel centers, an adjacency list, an adjacency matrix, and superpixel properties.

The required packages can be found in requirements.txt.
```
conda create --name eval_metrics
conda activate eval_metrics
/anaconda/envs/eval_metrics/bin/pip install -r requirements.txt
```

***

## I. Berkeley Segmentation Dataset 500

Metric validation experiments were conducted on the 200 images in the BSDS500 training set. To access these images, please proceed to the [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) website.

    Contour Detection and Hierarchical Image Segmentation
    P. Arbelaez, M. Maire, C. Fowlkes and J. Malik.
    IEEE TPAMI, Vol. 33, No. 5, pp. 898-916, May 2011.

***

## II. Functions

```python
def get_K(labels)
```
Inputs:
- labels = n x 1 where n is the number of pixels

Outputs: 
- K = number of superpixels

```python
def get_n(img)
```
Inputs:
- img = n x d where n is the number of pixels and d is the number of feature dimensions

Outputs:
- n = number of pixels in image

```python 
def ComputeAdjacency(labels, connectivity) 
```
Inputs: 
- labels = R x C where R is the number of rows in an image and C is the number #of columns
- connectivity = 4 or 8

Outputs:
- Am = adjacency matrix, K x K where K is the number of superpixels
- Al = adjacency list of length K where K is the number of superpixels

```python
def ComputeCenters(img, labels)
 ```
Inputs: 
- img = n x d where n is the number of pixels in the image and d is the dimensionality of a pixel
- labels = n x 1 where n is the number of pixels in the image

Output:
- centers = K x d where K is the number of superpixels and d is the dimensionality of a pixel

```python
def CalinskiHarabasz(img, labels) 
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- score = metric score

```python
def LocalCalinskiHarabasz(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- kscores = metric scores for each superpixel, length K

```python
 def LocalCalinskiHarabasz2(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- kscores = metric scores for each superpixel, length K

```python
def DaviesBouldin(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- score = metric score

```python
def LocalDaviesBouldin(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- kscores = metric scores for each superpixel, length K

```python
def Dunn(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- score = metric score

```python
def LocalDunn(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- kscores = metric scores for each superpixel, length K

```python
def I(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- score = metric score

```python 
def LocalI(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- kscores = metric scores for each superpixel, length K

```python
def RSquared(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- score = metric score

```python
def LocalRSquared(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- kscores = metric scores for each superpixel, length K

```python
def RootMeanSquaredStandardDeviation(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- score = metric score

```python
def LocalRootMeanSquaredStandardDeviation(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- kscores = metric scores for each superpixel, length K

```python
def XieBeni(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- score = metric score

```python
def LocalXieBeni(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- kscores = metric scores for each superpixel, length K

```python
def Variance(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- kscores = metric scores for each superpixel, length K

```python
def Compactness(img, labels)
```
Inputs: 
- img = R x C x d where R is the number of rows in an image and C is the number #of columns and d is the dimensionality of a pixel
- labels = R x C where where R is the number of rows in an image and C is the number #of columns

Output: 
- score = metric score

***

## III. Demo

`demo.ipynb` includes a demonstration using the image displayed below to obtain superpixels and run the 17 metrics is included.

<img src = "alligator.jpg" width = "250" height = "250"/>

***

## IV. Citing

If you use this code, please cite ..

This code uses the function regionprops from [scikit image](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).
