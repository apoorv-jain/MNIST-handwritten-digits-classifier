# MNIST-handwritten-digits-classifier
An implementation of neural networks to classify handwritten digits as numerical values. We have used MNIST dataset to train the Neural Network(NN) model with 60,000 training examples for 300 epochs by using categorical cross entropy and softmax in the output layer. The model also contains a hidden layer which uses Relu activation function.
The model is implemented by using [Keras](HTTPS://keras.io) Scikit library and using [Tensorflow](https://tensorflow.org) in the backend.
The project contains the test and train files in the form of .csv format.There is also a provision for training the model and saving it in train.py as a model-h5 format file and use it later and test it in draw.py which uses open-cv and matplotlib to allow to draw a handwritten digit and let the computer to predict the correct value.

The changes and advances to be done:-
- Using CNNs(Convolution Neural Networks)

This could give better performance for the digits drawn by the user.

The files required to train and test can be downloaded from the following link:-https://www.kaggle.com/oddrationale/mnist-in-csv

The number of epochs, learning rate and other properties can be changed and worked out.
