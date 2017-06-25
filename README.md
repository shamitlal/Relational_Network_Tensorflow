# Relational_Network_Tensorflow
Tensorflow implementation of Relation Networks - <a href="https://arxiv.org/pdf/1706.01427.pdf"> A simple neural network module for relational reasoning</a>

The work currently supports only Sort-of-CLEVR task. <br>
It's a tensorflow implementation of the following <a href="https://github.com/kimhc6028/relational-networks/blob/master/README.md#sort-of-clevr">Pytorch implementation</a>

# RELATIONAL NETWORKS

An RN is a neural network module with a structure primed for relational reasoning. The design philosophy behind RNs is to constrain the functional form of a neural network so that it captures the core common properties of relational reasoning. In other words, the capacity to compute relations is baked into the RN architecture without needing to be learned, just as the capacity to reason about spatial, translation invariant properties is built-in to CNNs, and the capacity to reason about sequential dependencies is built into recurrent neural networks.

<img src="https://github.com/shamitlal/Relational_Network_Tensorflow/blob/master/figures/rn.png">

# Sort-of-CLEVR

Sort-of-CLEVR is a dataset similar to CLEVR. This dataset separates relational and non-relational questions. Sort-of-CLEVR consists of images of 2D colored shapes along with questions and answers about the images. Each image has a total of 6 objects, where each object is a randomly chosen shape (square or circle). 6 colors are used (red, blue, green, orange, yellow, gray) to unambiguously identify each object. Questions are hard-coded as fixed-length binary strings to reduce the difficulty involved with natural language question-word processing, and thereby remove any confounding difficulty with language parsing. For each image the authors generated 10 relational questions and 10 non-relational questions. <br>

Non-relational questions are composed of 3 subtypes:
<ol>
<li>Shape of certain colored object</li>
<li>Horizontal location of certain colored object : whether it is on the left side of the image or right side of the image</li>
<li>Vertical location of certain colored object : whether it is on the upside of the image or downside of the image</li>
</ol>

Relational questions are composed of 3 subtypes:
<ol>
<li>Shape of the object which is closest to the certain colored object</li>
<li>Shape of the object which is furthest to the certain colored object</li>
<li>Number of objects which have the same shape with the certain colored object</li>
</ol>
<br>
Questions are encoded into a vector of size of 11 : 6 for one-hot vector for certain color among 6 colors, 2 for one-hot vector of relational/non-relational questions. 3 for one-hot vector of 3 subtypes.
<br>
Some sample images of sort-of-CLEVR : <br>
<img src="https://github.com/shamitlal/Relational_Network_Tensorflow/blob/master/figures/soc1.png" width="256" height="256">
<img src="https://github.com/shamitlal/Relational_Network_Tensorflow/blob/master/figures/soc2.png" width="256" height="256">

# REQUIREMENTS
<ul>
<li>Python 2.7</li>
<li>numpy</li>
<li>tensorflow</li>
<li>opencv</li>
</ul>

# USAGE

Run following to generate sort-of-CLEVR dataset run:
```
$ python sort_of_clevr_generator.py
```
followed by 
```
$ python main.py
```
to train
