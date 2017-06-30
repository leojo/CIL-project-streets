from __future__ import absolute_import
from __future__ import division

import time
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image

IMG_PATCH_SIZE = 128
NUM_IMAGES = 100
NUM_SAMPLES = 100
NUM_EPOCHS = 50
BATCH_SIZE = 10
LEARNING_RATE = 1e-4

SAVE_FILE = "model.ckpt"
LOAD = True

def readable_time(seconds):
	secs = int(seconds)%60
	mins = int(seconds/60)%60
	hrs = int(seconds/3600)
	return hrs,mins,secs

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

def img_sample(imgs,labs,num):
	img_samples = []
	lab_samples = []
	normal_prob = np.asarray([0.04799823,  0.07454103,  0.10373811,  0.12928556,  0.14443707,0.14443707,  0.12928556,  0.10373811,  0.07454103,  0.04799823])
	target_nums = np.round((np.ones(10)*num*normal_prob)).astype(np.int32)
	num_per_ratio = [0]*10
	n,w,h,c = imgs.shape
	for i in range(np.sum(target_nums)):
		m = np.random.randint(n)
		x = np.random.randint(w-IMG_PATCH_SIZE)
		y = np.random.randint(h-IMG_PATCH_SIZE)
		ratio = np.sum(labs[m,x:x+IMG_PATCH_SIZE,y:y+IMG_PATCH_SIZE,1]).astype(np.float32)/(IMG_PATCH_SIZE*IMG_PATCH_SIZE)
		ratio_cat = min(int(ratio*10),9)
		while num_per_ratio[ratio_cat]>=target_nums[ratio_cat]:
			m = np.random.randint(n)
			x = np.random.randint(w-IMG_PATCH_SIZE)
			y = np.random.randint(h-IMG_PATCH_SIZE)
			ratio = np.sum(labs[m,x:x+IMG_PATCH_SIZE,y:y+IMG_PATCH_SIZE,1]).astype(np.float32)/(IMG_PATCH_SIZE*IMG_PATCH_SIZE)
			ratio_cat = min(int(ratio*10),9)
		num_per_ratio[ratio_cat] += 1
		img_samples.append(imgs[m,x:x+IMG_PATCH_SIZE,y:y+IMG_PATCH_SIZE,:])
		lab_samples.append(labs[m,x:x+IMG_PATCH_SIZE,y:y+IMG_PATCH_SIZE,:])
	print("Acheived the following distribution of training samples:")
	print(num_per_ratio)
	return np.asarray(img_samples),np.asarray(lab_samples)








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
		imgs.append(img)
		lab = np.round(mpimg.imread(label_name))
		lab = np.stack([1-lab,lab],axis=-1)
		labs.append(lab)
	return np.asarray(imgs), np.asarray(labs)

def load_test_image(file_path):
	img = mpimg.imread(file_path)
	return np.asarray([img])

def balance_train_data(imgs,labs,num_per_ratio):
	print("Balancing training data, randomly sampling until %d samples of each type have been generated"%(num_per_ratio))
	imgs_sampled, labs_sampled = img_sample(imgs,labs,num_per_ratio)
	indices = []
	ratios = []
	for i, lab in enumerate(labs_sampled):
		road_pixels = np.sum(lab[:,:,1]).astype(np.float32) #number of pixels of class 1 (road)
		tot_pixels = lab.shape[0]*lab.shape[1]
		ratio = road_pixels/tot_pixels
		ratios.append(ratio)
	ratios = np.asarray(ratios)
	hist = np.histogram(ratios,range=(0.0,1.0))
	print("Done balancing!")
	return imgs_sampled, labs_sampled





def unpool(value):
	#return tf.image.resize_images(value,tf.shape(target)[1:3])
	return tf.layers.conv2d_transpose(value,value.get_shape().as_list()[-1],2,2)

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
		scaling = 1
	else:
		scaling = 255
	Image.fromarray((img*scaling).astype(np.uint8)).show()


def save(img,name):
	if img.dtype == np.uint8:
		scaling = 1
	else:
		scaling = 255
	Image.fromarray((img*scaling).astype(np.uint8)).save(name)

def main():
	x = tf.placeholder(tf.float32, [None, None, None, 3])
	y_ = tf.placeholder(tf.float32, [None, None, None, 2])
	training = tf.placeholder(tf.bool)


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

	unpool4 = unpool(pool4)
	deconv4 = deconv(unpool4,64,7)

	unpool3 = unpool(deconv4)
	deconv3 = deconv(unpool3,64,7)

	unpool2 = unpool(deconv3)
	deconv2 = deconv(unpool2,64,7)

	unpool1 = unpool(deconv2)
	deconv1 = deconv(unpool1,64,7)

	predictions = conv(deconv1,2,1)

	loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = predictions)

	#precision, _ = tf.metrics.precision(y_,predictions)
	#recall, _ = tf.metrics.recall(y_,predictions)
	#score = tf.scalar_mult(2,tf.divide(tf.scalar_mult(precision,recall),tf.add(precision,recall)))

	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

	with tf.Session() as sess:
		saver = tf.train.Saver()
		imgs_original, labs_original = load_train_data("../data/training_smooth/", num_images=NUM_IMAGES, mode="jpg")
		if LOAD:
			saver.restore(sess,SAVE_FILE)
		else:
			sess.run(tf.global_variables_initializer())

			imgs, labs = balance_train_data(imgs_original,labs_original,NUM_SAMPLES)
			n_training = imgs.shape[0]

			training_indices = range(imgs.shape[0])

			training_times = []
			for e in range(NUM_EPOCHS):
				training_indices = np.random.permutation(training_indices)
				print("Training epoch %d of %d"%(e+1,NUM_EPOCHS))
				minibatches_per_epoch = int(np.ceil(n_training/BATCH_SIZE))
				for i in range(0,n_training,BATCH_SIZE):
					start_time = time.time()
					batch_range = training_indices[i:i+BATCH_SIZE]
					batch_number = int(i/BATCH_SIZE)+1
					print("Minibatch %d of %d. "%(batch_number,minibatches_per_epoch)),
					batch = [imgs[batch_range],labs[batch_range]]
					_, mean_loss = sess.run([train_step,tf.reduce_mean(loss)],feed_dict={x: batch[0], y_: batch[1], training: True})
					print("Loss = %f.\t"%mean_loss),
					training_times.append(time.time()-start_time)
					avg_training_time = np.mean(training_times)
					time_remaining = avg_training_time*(((NUM_EPOCHS-e-1)*minibatches_per_epoch)+(minibatches_per_epoch-batch_number))
					h, m, s = readable_time(time_remaining)
					print("[Estimated time remaining %.2d:%.2d:%.2d]\r"%(h,m,s))
				saver.save(sess,SAVE_FILE)

			print("Done training")
			succesful_save = saver.save(sess,SAVE_FILE)
			print("Model stored in %s"%succesful_save)


		if LOAD:
			print("Generating submission masks...")
			for i in range(1,51):
				img_batch = load_test_image("../data/test_set_smooth/test_%d/test_%d.jpg"%(i,i))
				print(img_batch.shape)
				img_prediction = tf.argmax(predictions,axis=-1).eval(feed_dict={x:img_batch, training: False})
				save(img_prediction[0],"submission_masks/test_%d_prediction.png"%i)
			print("Done!")

		else:
			print("Generating train predictions...")
			test_indices = range(10)
			test_imgs = imgs_original[test_indices]
			test_labs = tf.argmax(labs_original[test_indices],axis=-1).eval()
			test_preds = tf.argmax(predictions,axis=-1).eval(feed_dict={x:test_imgs, training: False})

			np.save("test_imgs.npy",test_imgs)
			np.save("test_labs.npy",test_labs)
			np.save("test_preds.npy",test_preds)

			for i in range(len(test_imgs)):
				name = "testImg%d"%(i)
				save(test_imgs[i],name+"_original.png")
				save(test_labs[i],name+"_groundtruth.png")
				save(test_preds[i],name+"_prediction.png")
			print("Done!")

		#print("Calculating score...")
		#precision, _ = tf.metrics.precision(test_labs,test_preds)
		#recall, _ = tf.metrics.recall(test_labs,test_preds)

		#score = 2*precision*recall/(precision+recall)
		#print(score)

if __name__ == '__main__':
		main()