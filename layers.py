from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    N = x.shape[0]
    modx = np.reshape(x,(N,-1))
    out = np.dot(modx,w)+b
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    db = np.sum(dout,axis=0)
    dx = np.dot(dout,w.T)
    dx = dx.reshape(x.shape)
    dw = np.dot(x.T,dout)
    dw = dw.reshape(w.shape,order='F')
    #dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0,x)
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    
    dx, x = None, cache
    dx = np.where(x>0,1,0)*dout
    
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x,axis = 0)
        xmu = x-sample_mean
        #Getting the variance
        sq = xmu ** 2
        sample_var = 1./N * np.sum(sq, axis = 0)
        sqrtvar = np.sqrt(sample_var + eps)
        invsqrvar = 1/sqrtvar
        xhat = xmu *invsqrvar
        gammax = np.multiply(xhat,gamma)
        out = gammax+beta
        cache = (mode,xhat,gamma,xmu,invsqrvar,sqrtvar,sample_var,eps)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        std = np.sqrt(running_var+eps)
        x_out_mod = (x-running_mean)/std
        out = np.multiply(x_out_mod,gamma)+beta
        cache = (mode, x, x_out_mod, gamma, beta, std)
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return out, cache

#https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    mode = cache[0]
    dx, dgamma, dbeta = None, None, None
    N,D = dout.shape
    if mode == 'train':
        mode,xhat,gamma,xmu,invsrqvar,sqrtvar,sample_var,eps = cache

        # multiply by 1
        dbeta = np.sum(dout, axis=0)
        dgammax = dout

        dgamma = np.sum(dgammax*xhat, axis=0)
        dxhat = dgammax * gamma

        dxmu1 = dxhat*invsrqvar
        dinvsqrvar = np.sum(dxhat*xmu, axis=0)

        dsqrtvar = -1. /(sqrtvar**2) * dinvsqrvar
        dsample_var = 0.5 * (1/np.sqrt(sample_var+eps))*dsqrtvar

        dsq = 1/N *np.ones((N,D))*dsample_var
        dxmu2 = 2*xmu *dsq

        dxmu = np.sum(dxmu1+dxmu2,axis=0)
        dx1 = dxmu1+dxmu2
        dmu = -1 * dxmu
        dx2 = 1. /N * np.ones((N,D)) * dmu
        dx = dx1+dx2
    if mode == 'test':
        mode, x, xn, gamma, beta, std = cache
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dxn = gamma * dout
        dx = dxn / std
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

#https://kevinzakka.github.io/2016/09/14/batch_normalization/
def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    N,D = dout.shape
    mode = cache[0]
    if mode == 'train':
        mode,xhat,gamma,xmu,invsrqvar,sqrtvar,sample_var,eps = cache
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout*xhat, axis=0)
        dxhat = dout * gamma
        dx = (1. / N) * invsrqvar * (N*dxhat - np.sum(dxhat, axis=0) - xhat*np.sum(dxhat*xhat, axis=0))
   
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    if mode == 'test':
        mode, x, xn, gamma, beta, std = cache
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dx = gamma * dout/std
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    N, D = x.shape
    sample_mean = np.mean(x,axis = 1)
    xmu = x.T-sample_mean
    #Getting the variance
    sq = xmu ** 2
    sample_var = 1./N * np.sum(sq, axis = 0)
    sqrtvar = np.sqrt(sample_var + eps)
    invsqrvar = 1/sqrtvar
    xhat = xmu *invsqrvar
    gammax = np.multiply(xhat.T,gamma)
    out = gammax+beta
    cache = (xhat,gamma,xmu,invsqrvar,sqrtvar,sample_var,eps)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    N,D = dout.shape
    xhat,gamma,xmu,invsrqvar,sqrtvar,sample_var,eps = cache
    
    # multiply by 1
    print (dout.shape)
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(xhat.T*dout, axis=0)
    dxhat = (gamma*dout)
    dx = (1. / N) * invsrqvar * (N*dxhat.T - np.sum(dxhat.T, axis=0) - xhat*np.sum(dxhat.T*xhat, axis=0))
    print (dx.shape)
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) > p)/p
        out = x* mask 
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        out = x
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout*mask
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    pad = conv_param['pad']
    stride = conv_param['stride']
    N,C,H,W =x.shape
    # N- number of records
    # c - channel
    # H - height and W- width
    npad = ((0, 0),(0,0), (pad,pad), (pad, pad))
    padded_x = np.pad(x,pad_width = npad,mode ='constant', constant_values=0)
    F,C,HH,WW = w.shape
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    N,C,H_new,W_new = padded_x.shape
    output = []
    counter = 0
    for i in range(N):
        for f in range(F):
            new_out = []
            for k in range(0,H_new-HH+1,stride):
                for l in range(0,W_new-WW+1,stride):
                    out1 = []
                    for j in range(C):
                        working_x = padded_x[i][j][k:k+HH,l:l+WW]
                        working_f = w[f][j][:][:]
                        out1.append(np.sum(working_x*working_f))
                        counter +=1
                    new_out.append(sum(out1)+b[f])
            inter_array = np.array(new_out)
            output.append(inter_array)
        
                    
                    
    # Stretching the vectors
    # F - number of filters /C- channel/HH- Height/WW- width
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    out1 = np.array(output)
    out1 =out1.ravel().reshape(N,F,H_out,W_out)
    out = out1 
    return out, cache


def conv_forward_naive1(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    pad = conv_param['pad']
    stride = conv_param['stride']
    N,C,H,W =x.shape
    #C,H,W = x.shape
    # N- number of records
    # c - channel
    # H - height and W- width
    npad = ((0, 0),(0,0), (pad,pad), (pad, pad))
    #padded_x = np.pad(x,pad_width = npad,mode ='constant', constant_values=0)
    F,C,HH,WW = w.shape
    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride
    #N,C,H_new,W_new = padded_x.shape
    output = []
    padded_x = x
    N,C,H_new,W_new = padded_x.shape
    print (b)
    for i in range(N):
        for f in range(F):
            new_out = []
            for k in range(0,H_new-stride,stride):
                for l in range(0,W_new-stride,stride):
                    out1 = []
                    for j in range(C):
                        #working_x = padded_x[i][j][k:k+HH,l:l+WW]
                        working_x = padded_x[j][k:k+HH,l:l+WW]
                        working_f = w[f][j][:][:]
                        out1.append(np.sum(working_x*working_f))
                    new_out.append(sum(out1)+b[f])
            new_array =np.array(new_out)
            new_array = new_array.reshape(3,3)
            output.append(new_array)
        
                    
                    
    # Stretching the vectors
    # F - number of filters /C- channel/HH- Height/WW- width
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    out = np.array(output)
    print (out)
    return out, cache

def conv_backward_naive1(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    Hp = 1 + (H + 2 * pad - HH) // stride
    Wp = 1 + (W + 2 * pad - WW) // stride
    npad = ((0, 0),(0,0), (pad,pad), (pad, pad))
    padded_x = np.pad(x,pad_width = npad,mode ='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_working = np.zeros_like(padded_x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    npad = ((0, 0),(0,0), (HH-1,WW-1), (HH-1, WW-1))
    padded_dout = np.pad(dout,pad_width = npad,mode ='constant', constant_values=0)
    N,C,H_new,W_new = padded_dout.shape
    #Calculation of dx
    for l in range(0,H_new-HH,stride):
        for k in range(0,W_new-WW,stride):
                        
                        working_dout = padded_dout[0][0][l:l+HH,k:k+WW]
                        working_w = w[0][0][:][:]
                        working_w = (np.rot90(working_w,2))
                        dx_working[0][0][l][k] = np.sum(working_w*working_dout)
    
    # Calculation of dw 
    for l in range(0,H_new-H-1,stride):
        for k in range(0,W_new-W-1,stride):
            working_dout = dout[0][0][:][:]
            working_x = padded_x[0][0][l:l+H,k:k+W]
            dw[0][0][l][k] = np.sum(working_dout*working_x)
    
    db = np.sum(dout)
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    dx = dx_working[0][0][HH-(HH-1):HH-(HH-1)+H,WW-(WW-1):WW-(WW-1)+W]
    return dx, dw, db

def conv_backward_naive2(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    Hp = 1 + (H + 2 * pad - HH) // stride
    Wp = 1 + (W + 2 * pad - WW) // stride
    npad = ((0, 0),(0,0), (pad,pad), (pad, pad))
    padded_x = np.pad(x,pad_width = npad,mode ='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_working = np.zeros_like(padded_x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    npad = ((0, 0),(0,0), (HH-1,WW-1), (HH-1, WW-1))
    padded_dout = np.pad(dout,pad_width = npad,mode ='constant', constant_values=0)
    N,F,H_new,W_new = padded_dout.shape
    N,F,Hout,Wout = dout.shape
    #Calculation of dx
    for i in range(N):
        for l in range(0,H_new-HH+1,stride):
            for k in range(0,W_new-WW+1,stride):
                for c in range(C):
                    for f in range(F):
                        working_dout = padded_dout[i][f][l:l+HH,k:k+WW]
                        working_w = w[f][c][:][:]
                        working_w = (np.rot90(working_w,2))
                        dx_working[i][c][l][k] = np.sum(working_w*working_dout)
    
    # Calculation of dw
    
        for l in range(0,H+(2*pad)-Hout+1,stride):
            for k in range(0,H+(2*pad)-Hout+1,stride):
                for c in range(C):
                    for f in range(F):
                        w_sum =0
                        for i in range(N):
                            working_dout = dout[i][f][:][:]
                            working_x = padded_x[i][c][l:l+H,k:k+W]
                            w_sum +=np.sum(working_dout*working_x)
                        dw[f][c][l][k] = w_sum

    
    db = np.sum(dout)
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    dx = dx_working[:,:,pad:-pad,pad:-pad]
    return dx, dw, db
def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    Hp = 1 + (H + 2 * pad - HH) // stride
    Wp = 1 + (W + 2 * pad - WW) // stride
    npad = ((0, 0),(0,0), (pad,pad), (pad, pad))
    padded_x = np.pad(x,pad_width = npad,mode ='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_working = np.zeros_like(padded_x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    npad = ((0, 0),(0,0), (HH-1,WW-1), (HH-1, WW-1))
    padded_dout = np.pad(dout,pad_width = npad,mode ='constant', constant_values=0)
    N,F,H_new,W_new = padded_dout.shape
    N,F,Hout,Wout = dout.shape
    #Calculation of dx
    for i in range(N):
        for l in range(0,H_new-HH+1,stride):
            for k in range(0,W_new-WW+1,stride):
                for c in range(C):
                    x_sum=0
                    for f in range(F):
                        working_dout = padded_dout[i][f][l:l+HH,k:k+WW]
                        working_w = w[f][c][:][:]
                        working_w = (np.rot90(working_w,2))
                        x_sum +=np.sum(working_w*working_dout)
                    dx_working[i][c][l][k] = x_sum
    
    # Calculation of dw
    
        for l in range(0,H+(2*pad)-Hout+1,stride):
            for k in range(0,H+(2*pad)-Hout+1,stride):
                for c in range(C):
                    for f in range(F):
                        w_sum =0
                        for i in range(N):
                            working_dout = dout[i][f][:][:]
                            working_x = padded_x[i][c][l:l+H,k:k+W]
                            w_sum +=np.sum(working_dout*working_x)
                        dw[f][c][l][k] = w_sum

    
    
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    dx = dx_working[:,:,pad:-pad,pad:-pad]
    db = np.sum(np.sum(np.sum(dout,axis =0),axis = 1),axis = 1)
    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    N,C,H,W = x.shape
    stride = pool_param['stride']
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    outlist = []
    index_list = []
    out = None
    for n in range(N):
        for c in range(C):
            for i in range(0,H-pool_height+1,stride):
                for j in range(0,W-pool_width+1,stride):
                    working_x = x[n][c][i:i+pool_height,j:j+pool_width]
                    outlist.append(np.max(working_x))
                    index_list.append(np.unravel_index(working_x.argmax(), working_x.shape))
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    out = np.reshape(outlist, (N,C,H_out,W_out))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x,index_list,pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x,index_list,pool_param = cache
    N,C,H,W = x.shape
    dx = np.zeros_like(x)
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N,C,H_new,W_new = dout.shape
    index_list.reverse()
    dout_list =dout.flatten().tolist()
    dout_list.reverse()
    for n in range(N):
        for c in range(C):
            for k in range(0,W-pool_width+1,pool_width):
                for l in range(0,H-pool_height+1,pool_height):
                    dout_to_be_mapped = dout_list.pop() 
                    index=index_list.pop()
                    working_dx = dx[n,c,k:k+pool_height,l:l+pool_width]
                    working_dx[index] = dout_to_be_mapped
                    dx[n,c,k:k+pool_width,l:l+pool_height]=working_dx
                    
                            
                    
                    
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    ###########################################################################
    #                           END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
