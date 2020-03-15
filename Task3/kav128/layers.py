import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    
    loss = reg_strength * np.linalg.norm(W) ** 2
    grad = reg_strength * 2 * W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    if len(predictions.shape) == 1:
        f = predictions.copy()
        f -= np.max(f)
        return np.exp(f) / np.sum(np.exp(f))
    else:
        f = predictions.copy()
        f -= np.max(f, axis=1).reshape((f.shape[0], -1))
        return np.exp(f) / np.sum(np.exp(f), axis=1).reshape((f.shape[0], -1))

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    if (type(target_index) == int):
        return -np.log(probs)[target_index]
    else:
        return -np.mean(np.log(probs[range(len(target_index)), target_index]))


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    
    sfmx = softmax(predictions)
    loss = cross_entropy_loss(sfmx, target_index)
    if type(target_index) == int:
        grad = sfmx
        grad[target_index] -= 1
        return loss, grad
    else:
        m = target_index.shape[0]
        grad = sfmx
        grad[range(m), target_index] -= 1
        return loss, grad / m



class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        
        self.multiplier = (X > 0)
        return X * self.multiplier
    
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """

        return d_out * self.multiplier

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.matmul(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        
        self.W.grad += np.matmul(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)
        d_input = np.matmul(d_out, self.W.value.T)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(np.random.randn(filter_size, filter_size, in_channels, out_channels))
        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        assert channels == self.in_channels

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        
        self.X = np.zeros((batch_size,
                           height + 2 * self.padding,
                           width + 2 * self.padding,
                           self.in_channels))
        self.X[:,
               self.padding : self.padding + height,
               self.padding : self.padding + width,
               :] = X
        X_cur = self.X.reshape(batch_size, -1)
        W_cur = self.W.value.reshape(-1, self.out_channels)

        result = np.empty((batch_size, out_height, out_width, self.out_channels))
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                cur_region = self.X[:,
                                    y : self.filter_size + y,
                                    x : self.filter_size + x,
                                    :].reshape(batch_size, -1)
                cur_dot = np.dot(cur_region, W_cur)
                result[:, y, x, :] = cur_dot + self.B.value
        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        result_grad = np.zeros((batch_size, height, width, channels))
        
        W_cur = self.W.value.reshape(-1, self.out_channels)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                cur_out = d_out[:, y, x, :]
                in_grad_cur = cur_out.dot(W_cur.T).reshape(batch_size, self.filter_size, self.filter_size, channels)
                result_grad[:,
                            y : y + self.filter_size,
                            x : x + self.filter_size,
                            :] += in_grad_cur
                
                cur_region = self.X[:,
                                    y : self.filter_size + y,
                                    x : self.filter_size + x,
                                    :].reshape(batch_size, -1)
                
                w_grad_cur = cur_region.T.dot(cur_out).reshape(self.filter_size,
                                                               self.filter_size,
                                                               self.in_channels,
                                                               self.out_channels)
                self.W.grad += w_grad_cur
                self.B.grad += np.sum(cur_out, axis=0)
    
        return result_grad[:,
                           self.padding : height - self.padding,
                           self.padding : width - self.padding,
                           :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        self.X = X
        result = np.empty((batch_size, out_height, out_width, channels))
        
        for y in range(out_height):
            for x in range(out_width):
                region = X[:,
                           y * self.stride : y * self.stride + self.pool_size,
                           x * self.stride : x * self.stride + self.pool_size,
                           :].reshape(batch_size, -1, channels)
                result[:, y, x, :] = region.max(axis=1)
    
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        result_grad = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                region = self.X[:,
                                y * self.stride : y * self.stride + self.pool_size,
                                x * self.stride : x * self.stride + self.pool_size,
                                :]
                
                maxx = region.reshape(batch_size, -1, channels).max(axis=1)[:, np.newaxis, np.newaxis, :]
                maxpos = (region == maxx)
                result_grad_region = result_grad[:,
                                                 y * self.stride : y * self.stride + self.pool_size,
                                                 x * self.stride : x * self.stride + self.pool_size,
                                                 :]
                
                cur_grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                result_grad_region += cur_grad * maxpos
        return result_grad

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = self.X_shape = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        result = X.reshape(batch_size, -1)
        return result

    def backward(self, d_out):
        # TODO: Implement backward pass
        result = d_out.reshape(self.X_shape)
        return result

    def params(self):
        # No params!
        return {}
