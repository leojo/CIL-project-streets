from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image

IMG_PATCH_SIZE = 50
NUM_EPOCHS = 1
BATCH_SIZE = 20

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

def unpool(value):
	"""N-dimensional version of the unpooling operation from
	https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

	:param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
	:return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
	"""
	sh = value.get_shape().as_list()
	dim = len(sh[1:-1])
	out = (tf.reshape(value, [-1] + sh[-dim:]))
	for i in range(dim, 0, -1):
			out = tf.concat(axis=i, values=[out, tf.zeros_like(out)])
	out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
	out = tf.reshape(out, out_size)
	return out

def pool(value):
	return tf.layers.max_pooling2d(inputs=value, pool_size=[2,2], strides=2)

def conv(value,filters,kernel,strides=1,activation=tf.nn.relu):
	return tf.layers.conv2d(
			inputs=value,
			filters=filters,
			kernel_size=kernel,
			strides=strides,
			padding="same",
			activity_regularizer=tf.layers.batch_normalization,
			activation=activation)

def deconv(value,filters,kernel,strides=1,activation=tf.nn.relu):
	return tf.layers.conv2d_transpose(
			inputs=value,
			filters=filters,
			kernel_size=kernel,
			strides=strides,
			padding="same",
			activity_regularizer=tf.layers.batch_normalization,
			activation=activation)

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

	# Network
	conv1 = conv(x,32,11)

	pool1 = pool(conv1)

	conv2 = conv(pool1,64,5)

	pool2 = pool(conv2)

	unpool1 = unpool(pool2)

	deconv1 = deconv(unpool1,64,5)

	unpool2 = unpool(deconv1)

	deconv2 = deconv(unpool2,32,11)

	# Dense Layer
	deconv2_flat = tf.contrib.layers.flatten(deconv2)
	dense = tf.layers.dense(inputs=deconv2_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=training)

	# Prediction Layer
	predictions = tf.layers.dense(inputs=dropout, units=IMG_PATCH_SIZE * IMG_PATCH_SIZE)

	#loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = logits)
	loss = tf.losses.hinge_loss(labels=y_, logits=predictions, reduction=tf.losses.Reduction.NONE)

	print_loss = tf.Print(loss,[loss])

	train_step = tf.train.AdamOptimizer(1e-4).minimize(print_loss)

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

		test_preds = predictions.eval(feed_dict={x:test_imgs, training: False})

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