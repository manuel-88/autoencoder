#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:00:25 2016

@author: iki
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from skimage.filters import threshold_otsu



#Load Street View House Numbers Dataset
mat = scipy.io.loadmat('manipulated_StreetView.mat')
mat = mat['train']


#Normalize Data

#Remove DC (mean of images)
mean_arr = np.array([np.mean(mat, axis = 1)]).T
mat = mat - mean_arr

#Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3*np.std(mat)
mat = (np.clip(mat, -pstd, pstd))/pstd

#Rescale from [-1,1] to [0.1,0.9]
mat = (mat +1) *0.4 + 0.1




#Thresholding the Images to binary 
###############################################################################
threshold_global_otsu = threshold_otsu(mat)
mat = mat >= threshold_global_otsu
###############################################################################





# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_input = 1024 #Street View data input (img shape: 32*32)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b1': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    return layer_1


# Building the decoder
def decoder(x):

    # Decoder Hidden layer with sigmoid activation #2
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    return layer_1

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mat.shape[0]/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = mat[i*batch_size:(i+1)*batch_size,:]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")



    # Applying encode and decode over test set    
    encode_decode = sess.run(
        y_pred, feed_dict={X: mat[:examples_to_show]})
    
        # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mat[i], (32, 32)), cmap="Greys_r")
        a[1][i].imshow(np.reshape(encode_decode[i], (32, 32)), cmap="Greys_r")
    f.show()
    plt.draw()
    