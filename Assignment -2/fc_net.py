from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros((hidden_dim))
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim,num_classes))
        self.params['b2'] = np.zeros((num_classes))
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        h_input,cache1 =affine_forward(X,self.params['W1'],self.params['b1'])
        h_output,cache2 = relu_forward(h_input)
        scores,cache3 = affine_forward(h_output,self.params['W2'],self.params['b2'])
        
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        data_loss,dloss = softmax_loss(scores, y)
        reg_loss = np.sum(0.5 * self.reg * (self.params['W1']**2)) + np.sum(0.5 * self.reg*(self.params['W2']**2))
        loss = data_loss+reg_loss
        grads ={}
        dh_output,grads['W2'],grads['b2'] =affine_backward(dloss,cache3)
        grads['W2'] += self.reg * self.params['W2']
        dh_input = relu_backward(dh_output,cache2)
        dX,grads['W1'],grads['b1'] = affine_backward(dh_input,cache1)
        grads['W1'] += self.reg * self.params['W1']
        #grads['b2'] = np.sum(dloss,axis=0)
        #grads['W2'] = np.dot(h_output.T,dloss) + (self.reg * self.params['W2'])
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers-1):
            weightkey = 'W'+str(i+1)
            biaskey = 'b'+str(i+1)
            #gammakey = 'gamma' + str(i+1)
            #betakey  = 'beta'+str(i+1)
            
            if (i==0):
                self.params[weightkey] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[i]))
                
            else:
                self.params[weightkey] = np.random.normal(0, weight_scale, (hidden_dims[i-1], hidden_dims[i]))
            self.params[biaskey] = np.zeros((hidden_dims[i]))
            #self.params[gammakey] = np.ones(hidden_dims[i])
            #self.params[betakey] = np.zeros(hidden_dims[i])
        # for the last layer
        self.params['W'+str(self.num_layers)] = np.random.normal(0, weight_scale, (hidden_dims[self.num_layers-2], num_classes))
        self.params['b'+str(self.num_layers)]=np.zeros((num_classes))
        #-- CHECKS----
        #print ("Number of Layers", self.num_layers)
        
        #print ("Number of Layers", self.num_layers)
        #print ("W1",self.params['W1'])
        #print ("b1",self.params['b1'])
        #print ("gamma1",self.params['gamma1'])
        #print ("beta1",self.params['beta1'])
        #print ("W2",self.params['W2'])
        #print ("b2",self.params['b2'])
        #print ("W3",self.params['W3'])
        #print ("b3",self.params['b3'])      
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        #self.bn_params = []
        if self.normalization=='batchnorm':
            print ("Using Batchnorm")
            self.bn_params = {'bn_param' + str(i + 1): {'mode': 'train',
                                                        'running_mean': np.zeros(dims[i + 1]),
                                                        'running_var': np.zeros(dims[i + 1])}
                              for i in range(len(dims) - 2)}
            gammas = {'gamma' + str(i + 1):
                      np.ones(dims[i + 1]) for i in range(len(dims) - 2)}
            betas = {'beta' + str(i + 1): np.zeros(dims[i + 1])
                     for i in range(len(dims) - 2)}

            self.params.update(betas)
            self.params.update(gammas)
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        #---- TO BE DELETED
        
        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params.items():
                bn_param[1]['mode'] = mode
        scores = None
        hidden_input ={}
        #cache_a ={}
        #cache_r ={}
        cache_history = []
        hidden_output ={}
        scores = X
        for i in range(self.num_layers-1):
                weightkey = 'W'+str(i+1)
                biaskey = 'b'+str(i+1)
                if self.normalization=='batchnorm':
                    gammakey = 'gamma'+str(i+1)
                    betakey = 'beta'+str(i+1)
                    bnkey = 'bn_param' + str(i+1)
                #hidden_input[i+1],cache_a[i+1] = affine_forward(hidden_input[i],self.params[weightkey],self.params[biaskey])
                #hidden_output[i+1],cache_r[i+1] = relu_forward(hidden_input[i+1])
                #scores,cache = affine_relu_forward(scores,self.params[weightkey],self.params[biaskey])
                    scores,cache = affine_batch_relu_forward(scores,self.params[weightkey],self.params[biaskey],self.params[gammakey],self.params[betakey],self.bn_params[bnkey])
                else:
                      scores,cache = affine_relu_forward(scores,self.params[weightkey],self.params[biaskey])
                cache_history.append(cache)
                
        #for the last layer
        #scores,cache_a[self.num_layers] = affine_forward(hidden_output[self.num_layers-1],self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
        scores,cache = affine_forward(scores,self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
        cache_history.append(cache)
        #softmax Loss
        data_loss,d_data_loss = softmax_loss(scores, y)
        d_reg_loss = {}
        reg_loss = 0
        
        #finding the regularisation loss and gradient for reg_loss
        for i in range(self.num_layers):
            reg_loss += np.sum(0.5 * self.reg * (self.params['W'+str(i+1)]**2))
            d_reg_loss[i+1] = self.reg*self.params['W'+str(i+1)]
        loss = data_loss+reg_loss
        grads ={}
        
        #for the last layer gradient
        dout,grads['W'+str(self.num_layers)],grads['b'+str(self.num_layers)] =affine_backward(d_data_loss,cache_history.pop())
        grads['W'+str(self.num_layers)] += self.reg * self.params['W'+str(self.num_layers)]                
        
        i = self.num_layers-1
        while i > 0:
            if self.normalization=='batchnorm':
            #dout, grads['W%d' % (i)], grads['b%d' % (i)] = affine_relu_backward(dout, cache_history.pop())
                dout, grads['W%d' % (i)], grads['b%d' % (i)],grads['gamma%d' %(i)],grads['beta%d' %(i)] = affine_batch_relu_backward(dout, cache_history.pop())
            else:
                 dout, grads['W%d' % (i)], grads['b%d' % (i)] = affine_relu_backward(dout, cache_history.pop())
            grads['W%d' % (i)] += d_reg_loss[i]
            i -= 1    
        
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        #loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def affine_batch_relu_forward(x, w, b,gamma,beta,params):
    """
    Convenience layer that perorms an affine transform followed by batchnorm followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma , beta,params : input to the batch norm layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    bn, bn_cache = batchnorm_forward(a, gamma, beta, params)
    out, relu_cache = relu_forward(bn)
    cache = (fc_cache,bn_cache, relu_cache)
    return out, cache

def affine_batch_relu_backward(dout,cache):

    fc_cache,bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx_bn,dgamma,dbeta = batchnorm_backward_alt(da,bn_cache)
    dx, dw, db = affine_backward(dx_bn, fc_cache)
    return dx, dw, db,dgamma,dbeta