# Deep Learning - CNN for Image Recognition
Classifier for images or video input using **Google TensorFlow**. 
Research project on Convolutional Neural Networks, feature extraction, transfer learning, adversary noise and *DeepDream*. 

In notebooks from 01 to 03 we mainly followed tutorials, with some changes and many observations, in order to learn how to use TensorFlow to build Convolutional Neural Networks for OCR and image recognition. Later notebooks follow our project specifications, while notebooks 06 and 07 were used for some more facoltative observations. 

The project was completed around February 2017.

## Keras
Keras subfolder contains more advanced experiments done with Keras. The first 4 notebooks are about MNIST, with single column and multi column CNNs and dataset expansion. Notebook 5 is an implementation of Transfer Learning using Keras with CIFAR-10.

## Libraries and credits
* Notebooks from 01 to 05 (included) use TensorFlow 0.9.0, Notebook 06 uses TensorFlow 0.11.head. 
* Notebooks 03 and 04 use PrettyTensor 0.6.2.
* Notebook 07 uses caffe library and some modified code from [google/deepdream](https://github.com/google/deepdream).
* Notebooks in Keras subfolder use Keras and TensorFlow 0.12.1
* Special thanks and credits to [Hvass-Labs](https://github.com/Hvass-Labs) for well made tutorials and libraries. 
* Notebooks were made on 3 different Docker machines running different environments, thanks to [jupyter/docker-stacks](https://github.com/jupyter/docker-stacks/tree/master/tensorflow-notebook) and [saturnism/deepdream-docker](https://github.com/saturnism/deepdream-docker).
