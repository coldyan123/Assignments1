import numpy as np
import math
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  train_num = X.shape[0]
  dim = X.shape[1]
  class_num = W.shape[1]
  for i in range(0, train_num):
      scores = X[i].dot(W)
      Sum = 0
      for j in range(0, class_num):
          Sum += math.exp(scores[j])
      for j in range(0, class_num):
          if j == y[i]:
              dW[:, j] += (math.exp(scores[y[i]]) / Sum - 1) * X[i]
          else:
              dW[:, j] += math.exp(scores[j]) / Sum * X[i]
      loss += -math.log(math.exp(scores[y[i]]) / Sum)    
  loss /= train_num
  dW /= train_num
  dW += 2 * reg * W
  
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  train_num = X.shape[0]
  dim = X.shape[1]
  class_num = W.shape[1]
  
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  tmp = exp_scores[range(0, train_num), y] / np.sum(exp_scores, axis=1)
  loss = np.sum(-np.log(tmp)) / train_num
  
  H = exp_scores / np.sum(exp_scores, axis=1).reshape((train_num, 1))
  H[range(0, train_num), y] -= 1
  dW = X.T.dot(H) / train_num + 2 * reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

