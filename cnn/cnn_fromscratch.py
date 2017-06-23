from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image

IMG_PATCH_SIZE = 200
NUM_EPOCHS = 1
BATCH_SIZE = 10

def img_crop(im, w, h):
		list_patches = []
		imgwidth = im.shape[0]
		imgheight = im.shape[1]
		is_2d = len(im.shape) < 3
		for i in range(0,imgheight,h):
				for j in range(0,imgwidth,w):
						if is_2d:
								im_patch = im[j:j+w, i:i+h]
						else:
								im_patch = im[j:j+w, i:i+h, :]
						list_patches.append(im_patch)
		return list_patches



def load_train_data(train_dir,patch_size,num_images=100):
	images_dir = train_dir+"images/"
	labels_dir = train_dir+"groundtruth/"
	imgs = []
	labs = []
	for i in range(1,num_images+1):
		imageid = "satImage_%.3d" % i
		image_name = images_dir+imageid+".png"
		label_name = labels_dir+imageid+".png"
		img = mpimg.imread(image_name)
		img_parts = img_crop(img,patch_size,patch_size)
		lab = mpimg.imread(label_name)
		lab_parts = img_crop(lab,patch_size,patch_size)
		for j in range(len(img_parts)):
			imgs.append(img_parts[j])
			labs.append(lab_parts[j])
	return np.asarray(imgs), np.asarray(labs).reshape((-1,patch_size*patch_size))


def show(img):
	if len(img.shape) < 2:
		Image.fromarray(((img.reshape((IMG_PATCH_SIZE,IMG_PATCH_SIZE)))*255).astype(np.uint8)).show()
	else:
		Image.fromarray((img*255).astype(np.uint8)).show()


def save(img,name):
	if len(img.shape) < 2:
		Image.fromarray(((img.reshape((IMG_PATCH_SIZE,IMG_PATCH_SIZE)))*255).astype(np.uint8)).save(name)
	else:
		Image.fromarray((img*255).astype(np.uint8)).save(name)

def main():
	x = tf.placeholder(tf.float32, [None, IMG_PATCH_SIZE, IMG_PATCH_SIZE, 3])
	y_ = tf.placeholder(tf.float32, [None, IMG_PATCH_SIZE*IMG_PATCH_SIZE])
	training = tf.placeholder(tf.bool)


	imgs, labs = load_train_data("../data/training/", IMG_PATCH_SIZE)
	n_training = imgs.shape[0]-BATCH_SIZE

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
			inputs=x,
			filters=96,
			kernel_size=[11, 11],
			strides = 4,
			padding="same",
			activation=tf.nn.relu)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=256,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


	# Convolutional Layer #3 and Pooling Layer #3
	conv3 = tf.layers.conv2d(
			inputs=pool2,
			filters=384,
			kernel_size=[3, 3],
			padding="same",
			activation=tf.nn.relu)

	conv4 = tf.layers.conv2d(
			inputs=conv3,
			filters=384,
			kernel_size=[3, 3],
			padding="same",
			activation=tf.nn.relu)

	conv5 = tf.layers.conv2d(
			inputs=conv4,
			filters=256,
			kernel_size=[3, 3],
			padding="same",
			activation=tf.nn.relu)

	pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

	# Dense Layer
	pool5_flat = tf.contrib.layers.flatten(pool5)
	dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
	dense2 = tf.layers.dense(inputs=dense1, units=4096, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense2, rate=0.4, training=training)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=IMG_PATCH_SIZE * IMG_PATCH_SIZE)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = logits))

	batch = tf.Variable(0)
	# Decay once per epoch, using an exponential schedule starting at 0.01.
	learning_rate = tf.train.exponential_decay(
			0.01,                # Base learning rate.
			batch * BATCH_SIZE,  # Current index into the dataset.
			n_training,          # Decay step.
			0.95,                # Decay rate.
			staircase=True)

	train_step = tf.train.MomentumOptimizer(learning_rate,0.0).minimize(loss,global_step=batch)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		random_order = np.random.permutation(range(imgs.shape[0]))
		training_indices = random_order[:n_training]
		test_indices = random_order[n_training:]

		for e in range(NUM_EPOCHS):
			training_indices = np.random.permutation(training_indices)
			print("Training epoch %d"%(e+1))
			for i in range(0,n_training,BATCH_SIZE):
				batch_range = training_indices[i:i+BATCH_SIZE]
				print("Training: %d of %d"%(int(i/BATCH_SIZE)+1,int(n_training/BATCH_SIZE)))
				batch = [imgs[batch_range],labs[batch_range]]
				train_step.run(feed_dict={x: batch[0], y_: batch[1], training: True})

		print("Done training")

		test_imgs = imgs[test_indices]
		test_labs = labs[test_indices]

		test_preds = logits.eval(feed_dict={x:test_imgs, training: False})

		np.save("test_imgs.npy",test_imgs)
		np.save("test_labs.npy",test_labs)
		np.save("test_preds.npy",test_preds)

		for i in range(len(test_imgs)):
			name = "testImg%d"%(i)
			save(test_imgs[i],name+"_original.png")
			save(test_labs[i],name+"_groundtruth.png")
			save(test_preds[i],name+"_prediction.png")

if __name__ == '__main__':
		main()