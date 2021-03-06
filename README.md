# Facial Emotion Detection on images

# Introduction

## Problem Description
We use a subset of the CK+ (expanded Cohn-Kanade)database, with 327 face images of 640x400 pixels across 123 subjects. Each image has a label from a set of 7 emotions:
**1=anger,2=contempt,  3=disgust,  4=fear,  5=happy,  6=sadness  and  7=surprise**. 
Our  goal  is  to  train  a  classifier  that, given a new image, automatically annotates it with the corresponding emotion.

## Approach
To build an image representation, we use Bag of Words (BoW) and Fisher Vector (FV) approaches. Both BoW and FV featurizations rely on a beforehand computation of local descriptors of images. We choose to consider the very popular SIFT (Scale-Invariant Feature Transform) descriptors. Very interesting properties of these descriptors with respect to our problem are their invariance to affine transformations, and robustness to changes in illumination, noise, and small changes in view point.

For those who are more comfortable learning through video, check out https://www.youtube.com/watch?v=NPcMS49V5hg For people who like reading, see this http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html#gsc.tab=0

Now with the idea to get more distinct and compact descriptors (Dimensionality Reduction), we perform a Principal Component Analysis (PCA) on the SIFT descriptors.To get more information on PCA, check out https://en.wikipedia.org/wiki/Principal_component_analysis

Bag of Visual Vectors approach is now applied using kMeans clustering to build the image representation and each image is now a histogram of these clusters

## Implementation Notes
 1. We are using the [Cohn-Kanade Facial Expression extended database](http://www.pitt.edu/~emotion/ck-spread.htm). We cannot distribute it so you will have to request it yourself
 2. Run ipython notebook to understand implementation of visual bag of words and fisher vector approack