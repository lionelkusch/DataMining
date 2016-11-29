"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# all tensorflow api is accessible through this
import tensorflow as tf
# to visualize the resutls
import matplotlib.pyplot as plt
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

# load data, 60K trainset and 10K testset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#functionActivation = tf.nn.sigmoid
functionActivation = tf.nn.relu

# 1. Define Variables and Placeholders
X = tf.placeholder(tf.float32, [None,784]) #the first dimension (None) will index the images
Y_= tf.placeholder(tf.float32, [None, 10])      # correct answers
# Weights initialised with small random values between -0.2 and +0.2
W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1)) # 784 = 28*28
B1 = tf.Variable(tf.zeros([200]))
W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
B2 = tf.Variable(tf.zeros([100]))
W3 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1))
B3 = tf.Variable(tf.zeros([60]))
W4 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1))
B4 = tf.Variable(tf.zeros([30]))
W5 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))
#learning rate placeholder
lr = tf.placeholder(tf.float32)
# placeholder for probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)
# 2. Define the model
Y1 = functionActivation(tf.matmul(X, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 = functionActivation(tf.matmul(Y1d, W2) + B2)
"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# all tensorflow api is accessible through this
import tensorflow as tf
# to visualize the resutls
import matplotlib.pyplot as plt
# 70k mnist dataset that comes with the tensorflow container
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

# load data, 60K trainset and 10K testset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#functionActivation = tf.nn.sigmoid
functionActivation = tf.nn.relu

# 1. Define Variables and Placeholders
X = tf.placeholder(tf.float32, [None,784]) #the first dimension (None) will index the images
Y_= tf.placeholder(tf.float32, [None, 10])      # correct answers
# Weights initialised with small random values between -0.2 and +0.2
W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1)) # 784 = 28*28
B1 = tf.Variable(tf.zeros([200]))
W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
B2 = tf.Variable(tf.zeros([100]))
W3 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1))
B3 = tf.Variable(tf.zeros([60]))
W4 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1))
B4 = tf.Variable(tf.zeros([30]))
W5 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))
#learning rate placeholder
lr = tf.placeholder(tf.float32)
# placeholder for probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)
# 2. Define the model
Y1 = functionActivation(tf.matmul(X, W1) + B1)
#For this small convolutional network, performance is actually nearly identical with and without dropout.
# Dropout is often very effective at reducing overfitting, but it is most useful when training very large neural networks.
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 = functionActivation(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)
Y3 = functionActivation(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)
Y4 = functionActivation(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.nn.log_softmax(tf.matmul(Y4d, W5) + B5)
Y = tf.exp(Ylogits)
# 3. Define the loss function
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(Ylogits, Y_) # calculate cross-entropy with logits
cross_entropy = tf.reduce_mean(cross_entropy)*100

# 4. Define the accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1)), tf.float32))

# 5. Define an optimizer
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

#
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.005
lr = tf.train.exponential_decay(starter_learning_rate,
                                global_step,
                                100,
                                0.96,
                                staircase=True)
# Passing global_step to minimize() will increment it at each step.
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy,global_step=global_step)



# initialize
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


def training_step(i, update_test_data, update_train_data):
    print( "\r", i,)
    ####### actual learning
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y,pkeep:0.75})

    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y,pkeep:1})
        train_a.append(a)
        train_c.append(c)


    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels,pkeep:1})
        test_a.append(a)
        test_c.append(c)


    return (train_a, train_c, test_a, test_c)


# 6. Train and test the model, store the accuracy and loss per iteration

train_a = []
train_c = []
test_a = []
test_c = []

training_iter = 10000
epoch_size = 100
for i in range(training_iter):
    test = False
    if i % epoch_size == 0:
        test = True
    a, c, ta, tc = training_step(i, test, test)
    train_a += a
    train_c += c
    test_a += ta
    test_c += tc

# 7. Plot and visualise the accuracy and loss

# accuracy training vs testing dataset
plt.plot(train_a)
plt.plot(test_a)
plt.grid(True)
plt.show()

# loss training vs testing dataset
plt.plot(train_c)
plt.plot(test_c)
plt.grid(True)
plt.show()

# Zoom in on the tail of the plots
zoom_point = 50

x_range = range(zoom_point,int(training_iter/epoch_size))

plt.plot(x_range, train_a[zoom_point:])
plt.plot(x_range, test_a[zoom_point:])
plt.grid(True)
plt.show()

plt.plot(train_c[zoom_point:])
plt.plot(test_c[zoom_point:])
plt.grid(True)
plt.show()
Y2d = tf.nn.dropout(Y2, pkeep)
Y3 = functionActivation(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)
Y4 = functionActivation(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.nn.log_softmax(tf.matmul(Y4d, W5) + B5)
Y = tf.exp(Ylogits)
# 3. Define the loss function
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(Ylogits, Y_) # calculate cross-entropy with logits
cross_entropy = tf.reduce_mean(cross_entropy)*100

# 4. Define the accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1)), tf.float32))

# 5. Define an optimizer
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

#
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.005
lr = tf.train.exponential_decay(starter_learning_rate,
                                global_step,
                                100,
                                0.96,
                                staircase=True)
# Passing global_step to minimize() will increment it at each step.
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy,global_step=global_step)



# initialize
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


def training_step(i, update_test_data, update_train_data):
    print( "\r", i,)
    ####### actual learning
    # reading batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)
    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y,pkeep:0.75})

    ####### evaluating model performance for printing purposes
    # evaluation used to later visualize how well you did at a particular time in the training
    train_a = []
    train_c = []
    test_a = []
    test_c = []
    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y,pkeep:1})
        train_a.append(a)
        train_c.append(c)


    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels,pkeep:1})
        test_a.append(a)
        test_c.append(c)


    return (train_a, train_c, test_a, test_c)


# 6. Train and test the model, store the accuracy and loss per iteration

train_a = []
train_c = []
test_a = []
test_c = []

training_iter = 10000
epoch_size = 100
for i in range(training_iter):
    test = False
    if i % epoch_size == 0:
        test = True
    a, c, ta, tc = training_step(i, test, test)
    train_a += a
    train_c += c
    test_a += ta
    test_c += tc

# 7. Plot and visualise the accuracy and loss

# accuracy training vs testing dataset
plt.plot(train_a)
plt.plot(test_a)
plt.grid(True)
plt.show()

# loss training vs testing dataset
plt.plot(train_c)
plt.plot(test_c)
plt.grid(True)
plt.show()

# Zoom in on the tail of the plots
zoom_point = 50

x_range = range(zoom_point,int(training_iter/epoch_size))

plt.plot(x_range, train_a[zoom_point:])
plt.plot(x_range, test_a[zoom_point:])
plt.grid(True)
plt.show()

plt.plot(train_c[zoom_point:])
plt.plot(test_c[zoom_point:])
plt.grid(True)
plt.show()