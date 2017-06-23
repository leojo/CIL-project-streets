from cnn_tut import deepnn, load_train_data, show
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image
imgs, labs = load_train_data("../data/training/")

x = tf.placeholder(tf.float32, [None, 100, 100, 3])
y_ = tf.placeholder(tf.float32, [None, 100*100])

y_conv, keep_prob = deepnn(x)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
