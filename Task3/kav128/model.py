import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net
    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network
        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.layers = [
            ConvolutionalLayer(
                in_channels=input_shape[2], 
                out_channels=conv1_channels,
                filter_size=3, 
                padding=1
            ),
            ReLULayer(),
            MaxPoolingLayer(pool_size=4, stride=4),
            ConvolutionalLayer(
                in_channels=conv1_channels, 
                out_channels=conv2_channels,
                filter_size=3, 
                padding=1
            ),
            ReLULayer(),
            MaxPoolingLayer(pool_size=4, stride=4),
            Flattener(),
            FullyConnectedLayer(
                n_input=(input_shape[0] * input_shape[1] * conv2_channels // 256), 
                n_output=n_output_classes
            ),
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for key, param in self.params().items():
            param.grad = np.zeros_like(param.value)

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        for layer in self.layers:
            X = layer.forward(X)
        
        loss, grad = softmax_with_cross_entropy(X, y)
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        for layer in self.layers:
            X = layer.forward(X)

        return np.argmax(X, axis=1)

    def params(self):
        # TODO: Aggregate all the params from all the layers
        # which have parameters
        result = {}
        
        for idx, layer in enumerate(self.layers):
            for key, param in layer.params().items():
                r_key = str(idx) + key
                result[r_key] = param

        return result