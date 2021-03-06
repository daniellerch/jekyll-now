---
layout: post
title: Steganalysis of Steghide
---

## Steganalysis of Steghide

### Introduction

To deal with LSB matching based methods we need heavy machinery. In this article we will see how to apply [machine learning](https://en.wikipedia.org/wiki/Machine_learning) to steganalysis).

1. [Introduction](#1-introduction)

2. [Data](#2-data)

3. [Feature extraction](#3-feature-extraction)

4. [Training the classifier](#4-training-the-classifier)

5. [Testing](#5-testing)


<br>
#### 1. Introduction

Machine learning was applied succesfuly to many applications and steganalysis is not an exception. To apply machine learning to steganalysis the first we need is a training database, that is a database of cover and stego images. Second we need a feature extractor, that is a program capable to extract data from the images that could be used to differentiate between cover and stego images. Finally we need a classifier. This classifier receives the features extracted from the cover and stego images and needs to know which images are cover and which images are stego. With this information the classifier will learn how to classify images into cover and stego. This method is not perfect but it can detect LSB matching with high accuracy under laboratory conditions. 

To keep things easy, in this article we are going to use [Aletheia](https://github.com/daniellerch/aletheia), a software to perform steganalysis. Please, download it.

<br>
#### 2. Data

We need a database of images to prepare and train our classifier. In the field of steganalysis is very common to use the [Bossbase](http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip), an image set of 10000 grayscale images with size 512x512. So, for this article we are going to use this dataset. Download and uncompress it.

As far as we want to prepare a detector, we need to prepare this detector to dectect a particular steganographic method. We are going to use HILL [[1](#references)] with a payload of 0.40. HILL is one of the most secure steganographic algorithms at the moment of writing this lines. So, we need to prepare also an image database with images that contains hidden information using HILL 0.40.

Let's suppose our cover images are in a folder named "bossbase". We can prepare a set of stego images using HILL 0.40 with the following command:

```bash
$ ./aletheia.py hill-sim bossbase 0.40 bossbase_hill040 
```

We will obtain a new folder "bossbase_hill040" with all the images in "bossbase" with data hidden using a HILL simulator.


<br>
#### 3. Feature extraction

Usually in machine learning with images we need to extract features from the images. These features are multidimensional data that represent some characteristics of the image. Ideally, these characteristics should be different when the images are cover and when they are stego.

During the last years, much research was destinated to find the right way to extract these features. You can find a lot of feature extractors [here](http://dde.binghamton.edu/download/feature_extractors/).

You can use these feature extractors with Matlab or Octave, or use Aletheia to keep things easy. With the following commands we can extract the features using a very good feature extractor called Rich Models [[2](#references)].

```bash
$ ./aletheia.py srm bossbase bossbase.fea 
$ ./aletheia.py srm bossbase_hill040 bossbase_hill040.fea
```

As a result we obtain files where every image is represented by a line with 34671 features. If the feature extractor is good enough these features will be different when the images are cover and when the images are stego.  


<br>
#### 4. Training the classifier

We have the data prepared and now we need to train a classifier.  There is a lot of different classifieres we can try but in steganalysis one of the most popular classifiers is the Ensemble Classifier [[3](#references)] based on LFD. Use it with Aletheia executing the following command:

```bash
$ ./aletheia.py e4s bossbase.fea bossbase_hill040.fea hill040.model
Validation score: 73.0

```

<br>
#### 5. Testing

If we are happy with the result obtained with our training set, we can consider our classifier prepared. To test one image we can use the following command:

```bash
$ ./aletheia.py e4s-predict hill040.model srm my_test_image.png
Stego, probability: 0.81

```
But as I said at the begining the results are good in laboratory conditions. That is, for example, the experiment we did. But in the real world the things are much harder because we have to deal with the Cover Source Mismatch problem or CSM. This is a problem produced by training with an incomplete dataset. Our database, no matter how big it is, does not contains a representation of all the types of images that exist. As a consequence, if the image we want to test is not well represented in our training set, the results are going to be wrong. 





<br>
### References

[1]. B. Li, M. Wang, J. Huang, and X. Li, A new cost function for spatial image steganography, in 2014 IEEE International Conference on Image Processing (ICIP), pp. 4206–4210, IEEE, 2014.

[2]. J. Fridrich and J. Kodovsky, Rich models for steganalysis of digital images, IEEE Transactions on Information Forensics and Security.

[3]. J. Kodovský, J. Fridrich, and V. Holub, Ensemble Classifiers for Steganalysis of Digital Media. IEEE Transactions on Information Forensics and Security, Vol. 7, No. 2, pp. 432-444, April 2012. 







