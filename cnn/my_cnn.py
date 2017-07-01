from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image

IMG_SAMPLE_SIZE = 64
NUM_IMG_SAMPLES = 20

NUM_EPOCHS = 30
BATCH_SIZE = 10

SAVE_PATH = "model_2ndEra.ckpt"

def readable_time(seconds):
	secs = int(seconds)%60
	mins = int(seconds/60)%60
	hrs = int(seconds/3600)
	return hrs,mins,secs

def load_train_data(train_dir,num_images=100,mode="png"):
	images_dir = train_dir+"images/"
	labels_dir = train_dir+"groundtruth/"
	imgs = []
	labs = []
	for i in range(1,num_images+1):
		imageid = "satImage_%.3d" % i
		image_name = images_dir+imageid+"."+mode
		label_name = labels_dir+imageid+".png"
		img = mpimg.imread(image_name)
		lab = np.round(mpimg.imread(label_name))
		lab = np.stack([1-lab,lab],axis=-1)
		imgs.append(img)
		labs.append(lab)
	return np.asarray(imgs), np.asarray(labs)

def balance_train_data(imgs, labs):
	print("Balancing training data...")
	num_per_ratio = [0]*10
	imgs_sampled = []
	labs_sampled = []
	n,w,h,c = imgs.shape
	for i in range(NUM_IMG_SAMPLES*10):
		while True:
			m = np.random.randint(n-IMG_SAMPLE_SIZE)
			x = np.random.randint(w-IMG_SAMPLE_SIZE)
			y = np.random.randint(h-IMG_SAMPLE_SIZE)
			img_sample = imgs[m,x:x+IMG_SAMPLE_SIZE,y:y+IMG_SAMPLE_SIZE,:]
			lab_sample = labs[m,x:x+IMG_SAMPLE_SIZE,y:y+IMG_SAMPLE_SIZE,:]
			ratio = np.sum(lab_sample[:,:,1]).astype(np.float32)/(IMG_SAMPLE_SIZE*IMG_SAMPLE_SIZE)
			ratio_index = min(int(ratio*10),9)
			if num_per_ratio[ratio_index] < NUM_IMG_SAMPLES:
				break
		num_per_ratio[ratio_index] += 1
		imgs_sampled.append(img_sample)
		labs_sampled.append(lab_sample)
	print("Done!")
	return np.asarray(imgs_sampled), np.asarray(labs_sampled)

#def load_test_data(test_dit,patch_size,num_images=50):


def unpool(value,target):
	#return tf.image.resize_images(value,tf.shape(target)[1:3])
	return tf.layers.conv2d_transpose(
		inputs=value,
		filters=value.get_shape()[3],
		kernel_size=2,
		strides=2,
		padding="same")

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
	if img.dtype == np.uint8:
		scale = 1
	else:
		scale = 255
	Image.fromarray((img*scale).astype(np.uint8)).show()


def save(img,name):
	if img.dtype == np.uint8:
		scale = 1
	else:
		scale = 255
	Image.fromarray((img*scale).astype(np.uint8)).save(name)

def main():
	imgs, labs = load_train_data("../data/training_smooth/", mode="jpg")
	imgs_sampled, labs_sampled = balance_train_data(imgs,labs)

	x = tf.placeholder(tf.float32, [None, None, None, 3])
	y_ = tf.placeholder(tf.float32, [None, None, None, 2])
	training = tf.placeholder(tf.bool)

	n_training = imgs_sampled.shape[0]

	# Network
	norm = tf.nn.lrn(x, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
								name='norm1')

	conv1 = conv(norm,64,7)
	pool1 = pool(conv1)

	conv2 = conv(pool1,64,7)
	pool2 = pool(conv2)

	conv3 = conv(pool2,64,7)
	pool3 = pool(conv3)

	conv4 = conv(pool3,64,7)
	pool4 = pool(conv4)

	unpool4 = unpool(pool4,conv4)
	deconv4 = deconv(unpool4,64,7)

	unpool3 = unpool(deconv4,conv3)
	deconv3 = deconv(unpool3,64,7)

	unpool2 = unpool(deconv3,conv2)
	deconv2 = deconv(unpool2,64,7)

	unpool1 = unpool(deconv2,conv1)
	deconv1 = deconv(unpool1,64,7)

	predictions = deconv(deconv1,2,1)

	loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = predictions)

	train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

	with tf.Session() as sess:
		saver = tf.train.Saver()

		sess.run(tf.global_variables_initializer())

		training_indices = np.random.permutation(range(n_training))
		test_indices = range(10)
		training_times = []
		start_time = time.time()
		steps_per_epoch = int(n_training/BATCH_SIZE)
		for e in range(NUM_EPOCHS):
			epochs_remaining = NUM_EPOCHS-e-1
			training_indices = np.random.permutation(training_indices)
			print("Training epoch %d of %d"%(e+1,NUM_EPOCHS))
			for i in range(0,n_training,BATCH_SIZE):
				current_step = int(i/BATCH_SIZE)+1
				batch_range = training_indices[i:i+BATCH_SIZE]
				print("Training: %d of %d"%(current_step,steps_per_epoch))
				batch = [imgs_sampled[batch_range],labs_sampled[batch_range]]
				_, mean_loss = sess.run([train_step,tf.reduce_mean(loss)],feed_dict={x: batch[0], y_: batch[1], training: True})
				print("Loss = %f"%mean_loss)
				end_time = time.time()
				training_times.append(end_time-start_time)
				start_time = end_time
				avg_step_time = np.mean(training_times)
				steps_remaining_this_epoch = steps_per_epoch-current_step
				steps_remaining = epochs_remaining*steps_per_epoch+steps_remaining_this_epoch
				time_remaining = avg_step_time*steps_remaining
				print("[----- Estimated time remaining %.2d:%.2d:%.2d -----]"%readable_time(time_remaining))

		print("Done training")
		success = saver.save(sess,SAVE_PATH)
		print("Model saved in %s"%(success))

		test_imgs = imgs[test_indices]
		test_labs = tf.argmax(labs[test_indices],axis=-1).eval()

		test_preds = tf.argmax(predictions,axis=-1).eval(feed_dict={x:test_imgs, training: False})

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