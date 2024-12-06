# Neural Networks

This is a single hidden layer perceptron in OOP style.
It is obviously not supposed to be very good, rather it
is an exercise in understanding the backpropagation algorithm.

The model accepts any activation for the hidden layer as a parameter, 
so long as it is something that autograd can differentiate.

The loss function is hardcoded to be the quadratic loss.

Extensions: 
- Everything happens by-sample. By-batch is better and should 
just be some matrix multiplication.
- Multiple hidden layers, with different activation functions.
- Stochastic gradient descent.
- Momentum in gradient descent?
- Regularization.


References:
  - http://neuralnetworksanddeeplearning.com/chap2.html