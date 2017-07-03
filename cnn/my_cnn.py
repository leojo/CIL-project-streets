from __future__ import absolute_import
from __future__ import division

import os
import sys
import time

import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image

IMG_SAMPLE_SIZE = 128
NUM_IMG_SAMPLES = 2000
SAMPLE_DIST = "normal" # Must be either "normal" or "uniform"

NUM_EPOCHS = 500
BATCH_SIZE = 50

LEARNING_RATE = 1e-4

LOAD = True
LOAD_DATA = False #Specifies wether to load the training data or resample
LOAD_PATH = "model_3rdEra.ckpt"
DATA_LOAD_PATH = "data_3rdEra.npy"
SAVE_PATH = "model_3rdEra.ckpt"
DATA_SAVE_PATH = "data_3rdEra.npy"

EPOCH_SCORE_FILE = "epoch_scores.npy"

SUBMISSION_PRED_BATCH_SIZE = 5
SUBMISSION_DIR = "submission_masks"

def readable_time(seconds):
	secs = int(seconds)%60
	mins = int(seconds/60)%60
	hrs = int(seconds/3600)
	return hrs,mins,secs

def f1_score(pred,lab):
	TP = np.sum(pred*lab,dtype=np.float64)
	TN = np.sum((1-pred)*(1-lab),dtype=np.float64)
	FP = np.sum(pred*(1-lab),dtype=np.float64)
	FN = np.sum((1-pred)*lab,dtype=np.float64)

	precision = TP/(TP+FP)
	recall = TP/(TP+FN)

	f1 = 2*(precision*recall)/(precision+recall)

	return f1

# Attempting to write f1_loss functin, hope this works.
def tf_f1_score(labels, predictions):
	TP,_ = tf.metrics.true_positives(labels,predictions)
	FP,_ = tf.metrics.false_positives(labels,predictions)
	FN,_ = tf.metrics.false_negatives(labels,predictions)

	precision = TP/(TP+FP)
	recall = TP/(TP+FN)

	f1_score = 2*(precision*recall)/(precision+recall)
	return f1_score




def img_crop(im, w, h):
	if w == -1 and h == -1:
		return [im]	
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

def load_test_data(test_dir,num_images=50,mode="png"):
	imgs = []
	for i in range(1,num_images+1):
		imageid = "test_%d" % i
		image_name = os.path.join(test_dir,imageid,imageid+"."+mode)
		img = mpimg.imread(image_name)
		imgs.append(img)
	return np.asarray(imgs)

def balance_train_data(imgs, labs):
	print("Balancing training data...")
	normal_prob = np.asarray([0.04799823,  0.07454103,  0.10373811,  0.12928556,  0.14443707,0.14443707,  0.12928556,  0.10373811,  0.07454103,  0.04799823])
	uniform = np.asarray([0.1]*10)
	if SAMPLE_DIST == "normal":
		dist = normal_prob
	elif SAMPLE_DIST == "uniform":
		dist = uniform
	else:
		print("SAMPLE_DIST must be either \"normal\" or \"uniform\"!")
		exit()
	target_nums = np.round((np.ones(10)*NUM_IMG_SAMPLES*dist)).astype(np.int32)
	num_per_ratio = [0]*10
	imgs_sampled = []
	labs_sampled = []
	n,w,h,c = imgs.shape
	for i in range(np.sum(target_nums)):
		while True:
			m = np.random.randint(n)
			x = np.random.randint(w-IMG_SAMPLE_SIZE)
			y = np.random.randint(h-IMG_SAMPLE_SIZE)
			img_sample = imgs[m,x:x+IMG_SAMPLE_SIZE,y:y+IMG_SAMPLE_SIZE,:]
			lab_sample = labs[m,x:x+IMG_SAMPLE_SIZE,y:y+IMG_SAMPLE_SIZE,:]
			ratio = np.sum(lab_sample[:,:,1]).astype(np.float32)/(IMG_SAMPLE_SIZE*IMG_SAMPLE_SIZE)
			ratio_index = min(int(ratio*10),9)
			if num_per_ratio[ratio_index] < target_nums[ratio_index]:
				break
		num_per_ratio[ratio_index] += 1
		imgs_sampled.append(img_sample)
		labs_sampled.append(lab_sample)
	print("Done! Achieved the following distribution of training samples:")
	print(num_per_ratio)
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
	x = tf.placeholder(tf.float32, [None, None, None, 3])
	y_ = tf.placeholder(tf.float32, [None, None, None, 2])
	training = tf.placeholder(tf.bool)

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

	pred_argmax = tf.argmax(predictions,axis=-1)

	score = tf.reduce_mean(tf_f1_score(labels=tf.argmax(y_,axis=-1), predictions=pred_argmax))

	loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = predictions)

	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

	epochs_done = tf.Variable(0)

	with tf.Session() as sess:
		saver = tf.train.Saver()
		imgs, labs = load_train_data("../data/training_smooth/", mode="jpg")
		if LOAD:
			saver.restore(sess,LOAD_PATH)
			print("Model restored!")
			resume_epoch = epochs_done.eval()
			#epoch_scores = np.load(EPOCH_SCORE_FILE)
			#assert len(epoch_scores) == resume_epoch
			if resume_epoch < NUM_EPOCHS:
				print("Resuming training from epoch %d"%(resume_epoch+1))
				if LOAD_DATA:
					imgs_sampled = np.load("imgs_"+DATA_LOAD_PATH)
					labs_sampled = np.load("labs_"+DATA_LOAD_PATH)
				else:
					imgs_sampled, labs_sampled = balance_train_data(imgs,labs)

		else:
			#epoch_scores = np.array([])
			imgs_sampled, labs_sampled = balance_train_data(imgs,labs)
			np.save("imgs_"+DATA_SAVE_PATH,imgs_sampled)
			np.save("labs_"+DATA_SAVE_PATH,labs_sampled)
			sess.run(tf.global_variables_initializer())
			resume_epoch = 0

		if resume_epoch < NUM_EPOCHS:
			n_training = imgs_sampled.shape[0]

			training_indices = np.random.permutation(range(n_training))
			training_times = []
			train_start = time.time()
			start_time = train_start
			steps_per_epoch = int(np.ceil((1.0*n_training)/BATCH_SIZE))
			for e in range(resume_epoch,NUM_EPOCHS):
				epochs_remaining = NUM_EPOCHS-e-1
				training_indices = np.random.permutation(training_indices)
				for i in range(0,n_training,BATCH_SIZE):
					current_step = int(i/BATCH_SIZE)+1
					batch_range = training_indices[i:i+BATCH_SIZE]
					print("Training: Epoch %.3d of %d, minibatch %.2d of %d. "%(e+1,NUM_EPOCHS,current_step,steps_per_epoch)),
					batch = [imgs_sampled[batch_range],labs_sampled[batch_range]]
					_, mean_loss = sess.run([train_step,tf.reduce_mean(loss)],feed_dict={x: batch[0], y_: batch[1], training: True})
					print("Loss = %f. "%mean_loss),
					end_time = time.time()
					training_times.append(end_time-start_time)
					start_time = end_time
					avg_step_time = np.mean(training_times)
					steps_remaining_this_epoch = steps_per_epoch-current_step
					steps_remaining = epochs_remaining*steps_per_epoch+steps_remaining_this_epoch
					time_remaining = avg_step_time*steps_remaining
					print("[----- Estimated time remaining %.2d:%.2d:%.2d -----]"%readable_time(time_remaining))
				saver.save(sess,SAVE_PATH)
				sess.run(epochs_done.assign_add(1))
				#current_epoch_scores = []
				#for i in range(0,100,10):
				#	test_indices = range(i,i+10)
				#	test_imgs = imgs[test_indices]
				#	test_labs = labs[test_indices]
				#	test_pred_score = sess.run(score,feed_dict={x:test_imgs, y_:test_labs, training: False})
				#	current_epoch_scores.append(test_pred_score)
					
				#score = np.mean(current_epoch_scores)
				#print("Epoch %d finished with F1-score: %f"%(e+1,score))
				#epoch_scores = np.append(epoch_scores,score)
				#np.save(EPOCH_SCORE_FILE,epoch_scores)
						

			print("Done training!")
			#np.save(EPOCH_SCORE_FILE,epoch_scores)
			print("Trained on %d samples for %d epochs with minibatch size %d"%(NUM_IMG_SAMPLES,NUM_EPOCHS-resume_epoch,BATCH_SIZE))
			print("Training took %.2d:%.2d:%.2d"%readable_time(time.time()-train_start))
			success = saver.save(sess,SAVE_PATH)
			print("Model saved in %s"%(success))


		for i in range(0,100,10):
			test_indices = range(i,i+10)
			test_imgs = imgs[test_indices]
			test_labs = tf.argmax(labs[test_indices],axis=-1).eval()

			test_preds = tf.argmax(predictions,axis=-1).eval(feed_dict={x:test_imgs, training: False})

			#np.save("test_imgs.npy",test_imgs)
			#np.save("test_labs.npy",test_labs)
			#np.save("test_preds.npy",test_preds)

			for j in range(len(test_imgs)):
				name = "satImage_%.3d"%(i+j+1)
				#save(test_imgs[i],name+"_original.png")
				#save(test_labs[i],name+"_groundtruth.png")
				save(test_preds[j],name+"_prediction.png")

		if resume_epoch == NUM_EPOCHS:
			submission_imgs = load_test_data("../data/test_set_smooth",mode="jpg")
			for i in range(0,50,SUBMISSION_PRED_BATCH_SIZE):
				submission_batch = submission_imgs[i:i+SUBMISSION_PRED_BATCH_SIZE]
				submission_preds = tf.argmax(predictions,axis=-1).eval(feed_dict={x:submission_batch, training: False})
				for j in range(SUBMISSION_PRED_BATCH_SIZE):
					name = "test_%d_mask.png"%(i+j+1)
					print("Saving "+name+" in "+SUBMISSION_DIR)
					save(submission_preds[j],os.path.join(SUBMISSION_DIR,name))


if __name__ == '__main__':
		main()