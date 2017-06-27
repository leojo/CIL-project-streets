# ========================================================================================== #
#																							 #
#																						     #
#																							 #
# ========================================================================================== #

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import code
import tensorflow.python.platform
import numpy
import tensorflow as tf
import datautils as DU

# Basically baseline program split to trainModel function and makePredictions function.
class SimpleFCN:

	NUM_CHANNELS = 3 # RGB images
	PIXEL_DEPTH = 255
	NUM_LABELS = 2
	TRAINING_SIZE = 20
	TEST_SIZE = 50
	VALIDATION_SIZE = 5  # Size of the validation set.
	SEED = 66478  # Set to None for random seed.
	BATCH_SIZE = 16 # 64
	NUM_EPOCHS = 5
	RECORDING_STEP = 1000
	# Set image patch size
	# IMG_PATCH_SIZE should be a multiple of 4
	# image size should be an integer multiple of this number!
	IMG_PATCH_SIZE = 4
	
	def __init__(self, sess, flags):
		self.session = sess
		self.FLAGS = flags

		# The variables below hold all the trainable weights. They are passed an
		# initial value which will be assigned when when we call:
		# {tf.initialize_all_variables().run()}
		self.conv1_weights = tf.Variable(
			tf.truncated_normal([5, 5, self.NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
								stddev=0.1,
								seed=self.SEED))
		self.conv1_biases = tf.Variable(tf.zeros([32]))
		self.conv2_weights = tf.Variable(
			tf.truncated_normal([5, 5, 32, 64],
								stddev=0.1,
								seed=self.SEED))
		self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
		self.fc1_weights = tf.Variable(  # fully connected, depth 512.
			tf.truncated_normal([int(self.IMG_PATCH_SIZE / 4 * self.IMG_PATCH_SIZE / 4 * 64), 512],
								stddev=0.1,
								seed=self.SEED))
		self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
		self.fc2_weights = tf.Variable(
			tf.truncated_normal([512, self.NUM_LABELS],
								stddev=0.1,
								seed=self.SEED))
		self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[self.NUM_LABELS]))

		self.saver = tf.train.Saver()




	def makePredictions(self, RESTORE_MODEL=False):

		# Restore the model
		try:
			self.saver.restore(self.session, self.FLAGS.train_dir + "/model.ckpt")
			print("Model restored.")
		except Exception, e:
			raise Exception("No model was found! You need to train a model before making predictions.")    	

		print ("Running prediction on test set")
		predictions_test_dir = "predictions_test/"
		train_data_filename = '../data/training/images/'
		if not os.path.isdir(predictions_test_dir):
			os.mkdir(predictions_test_dir)
		for i in range(1, self.TEST_SIZE+1):
			pimg = (self.get_prediction_masks(train_data_filename, i)*255).astype(numpy.uint8)
			Image.fromarray(pimg).save(predictions_test_dir + "mask_" + str(i) + ".png")     


	def trainModel(self, train_data, train_labels):
		train_data, train_labels, train_size = self.balanceTrainData(train_data, train_labels)

		# This is where training samples and labels are fed to the graph.
		# These placeholder nodes will be fed a batch of training data at each
		# training step using the {feed_dict} argument to the Run() call below.
		train_data_node = tf.placeholder( tf.float32, 
											shape=(self.BATCH_SIZE, self.IMG_PATCH_SIZE, self.IMG_PATCH_SIZE, self.NUM_CHANNELS))
		train_labels_node = tf.placeholder(tf.float32,
											shape=(self.BATCH_SIZE, self.NUM_LABELS))
		train_all_data_node = tf.constant(train_data)
		
		# Training computation: logits + cross-entropy loss.
		logits = self.model(train_data_node, True) # BATCH_SIZE*NUM_LABELS
		# print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=logits, labels=train_labels_node))
		tf.summary.scalar('loss', loss)

		all_params_node = [self.conv1_weights, self.conv1_biases, self.conv2_weights, self.conv2_biases, self.fc1_weights, self.fc1_biases, self.fc2_weights, self.fc2_biases]
		all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases']
		all_grads_node = tf.gradients(loss, all_params_node)
		all_grad_norms_node = []
		for i in range(0, len(all_grads_node)):
			norm_grad_i = tf.global_norm([all_grads_node[i]])
			all_grad_norms_node.append(norm_grad_i)
			tf.summary.scalar(all_params_names[i], norm_grad_i)
		
		# L2 regularization for the fully connected parameters.
		regularizers = (tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
						tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))
		# Add the regularization term to the loss.
		loss += 5e-4 * regularizers

		# Optimizer: set up a variable that's incremented once per batch and
		# controls the learning rate decay.
		batch = tf.Variable(0)
		# Decay once per epoch, using an exponential schedule starting at 0.01.
		learning_rate = tf.train.exponential_decay(
			0.01,                # Base learning rate.
			batch * self.BATCH_SIZE,  # Current index into the dataset.
			train_size,          # Decay step.
			0.95,                # Decay rate.
			staircase=True)
		tf.summary.scalar('learning_rate', learning_rate)
		
		# Use simple momentum for the optimization.
		optimizer = tf.train.MomentumOptimizer(learning_rate,
											   0.0).minimize(loss,
															 global_step=batch)

		# Predictions for the minibatch, validation set and test set.
		train_prediction = tf.nn.softmax(logits)
		# We'll compute them only once in a while by calling their {eval()} method.
		train_all_prediction = tf.nn.softmax(self.model(train_all_data_node))

		# Run all the initializers to prepare the trainable parameters.
		tf.global_variables_initializer().run(session=self.session)

		# Build the summary operation based on the TF collection of Summaries.
		summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir,
												graph=self.session.graph)
		print ('Initialized!')
		# Loop through training steps.
		print ('Total number of iterations = ' + str(int(self.NUM_EPOCHS * train_size / self.BATCH_SIZE)))

		training_indices = range(train_size)

		for iepoch in range(self.NUM_EPOCHS):

			# Permute training indices
			perm_indices = numpy.random.permutation(training_indices)

			for step in range (int(train_size / self.BATCH_SIZE)):

				offset = (step * self.BATCH_SIZE) % (train_size - self.BATCH_SIZE)
				batch_indices = perm_indices[offset:(offset + self.BATCH_SIZE)]

				# Compute the offset of the current minibatch in the data.
				# Note that we could use better randomization across epochs.
				batch_data = train_data[batch_indices, :, :, :]
				batch_labels = train_labels[batch_indices]
				# This dictionary maps the batch data (as a numpy array) to the
				# node in the graph is should be fed to.
				feed_dict = {train_data_node: batch_data,
							 train_labels_node: batch_labels}

				if step % self.RECORDING_STEP == 0:

					summary_str, _, l, lr, predictions = self.session.run(
						[summary_op, optimizer, loss, learning_rate, train_prediction],
						feed_dict=feed_dict)
					#summary_str = s.run(summary_op, feed_dict=feed_dict)
					summary_writer.add_summary(summary_str, step)
					summary_writer.flush()

					# print_predictions(predictions, batch_labels)

					print ('Epoch %.2f' % (float(step) * self.BATCH_SIZE / train_size))
					print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
					print ('Minibatch error: %.1f%%' % DU.error_rate(predictions,
																 batch_labels))

					sys.stdout.flush()
				else:
					# Run the graph and fetch some of the nodes.
					_, l, lr, predictions = self.session.run(
						[optimizer, loss, learning_rate, train_prediction],
						feed_dict=feed_dict)

			# Save the variables to disk.
			save_path = self.saver.save(self.session, self.FLAGS.train_dir + "/model.ckpt")
			print("Model saved in file: %s" % save_path)

		train_data_filename = '../data/training/images/'
		print ("Running prediction on training set")
		prediction_training_dir = "predictions_training/"
		if not os.path.isdir(prediction_training_dir):
			os.mkdir(prediction_training_dir)
		for i in range(1, self.TRAINING_SIZE+1):
			pimg = self.get_prediction_with_groundtruth(train_data_filename, i)
			Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
			oimg = self.get_prediction_with_overlay(train_data_filename, i)
			oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")    

	# We will replicate the model structure for the training subgraph, as well
	# as the evaluation subgraphs, while sharing the trainable parameters.
	def model(self, data, train=False):
		"""The Model definition."""
		# 2D convolution, with 'SAME' padding (i.e. the output feature map has
		# the same size as the input). Note that {strides} is a 4D array whose
		# shape matches the data layout: [image index, y, x, depth].
		conv = tf.nn.conv2d(data,
							self.conv1_weights,
							strides=[1, 1, 1, 1],
							padding='SAME')
		# Bias and rectified linear non-linearity.
		relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
		# Max pooling. The kernel size spec {ksize} also follows the layout of
		# the data. Here we have a pooling window of 2, and a stride of 2.
		pool = tf.nn.max_pool(relu,
							  ksize=[1, 2, 2, 1],
							  strides=[1, 2, 2, 1],
							  padding='SAME')

		conv2 = tf.nn.conv2d(pool,
							self.conv2_weights,
							strides=[1, 1, 1, 1],
							padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_biases))
		pool2 = tf.nn.max_pool(relu2,
							  ksize=[1, 2, 2, 1],
							  strides=[1, 2, 2, 1],
							  padding='SAME')

		# Uncomment these lines to check the size of each layer
		# print 'data ' + str(data.get_shape())
		# print 'conv ' + str(conv.get_shape())
		# print 'relu ' + str(relu.get_shape())
		# print 'pool ' + str(pool.get_shape())
		# print 'pool2 ' + str(pool2.get_shape())


		# Reshape the feature map cuboid into a 2D matrix to feed it to the
		# fully connected layers.
		pool_shape = pool2.get_shape().as_list()
		reshape = tf.reshape(
			pool2,
			[pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
		# Fully connected layer. Note that the '+' operation automatically
		# broadcasts the biases.
		hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
		# Add a 50% dropout during training only. Dropout also scales
		# activations such that no rescaling is needed at evaluation time.
		#if train:
		#    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
		out = tf.matmul(hidden, self.fc2_weights) + self.fc2_biases

		if train == True:
			summary_id = '_0'
			s_data = self.get_image_summary(data)
			filter_summary0 = tf.summary.image('summary_data' + summary_id, s_data)
			s_conv = self.get_image_summary(conv)
			filter_summary2 = tf.summary.image('summary_conv' + summary_id, s_conv)
			s_pool = self.get_image_summary(pool)
			filter_summary3 = tf.summary.image('summary_pool' + summary_id, s_pool)
			s_conv2 = self.get_image_summary(conv2)
			filter_summary4 = tf.summary.image('summary_conv2' + summary_id, s_conv2)
			s_pool2 = self.get_image_summary(pool2)
			filter_summary5 = tf.summary.image('summary_pool2' + summary_id, s_pool2)

		return out

	def balanceTrainData(self, train_data, train_labels):
		c0 = 0
		c1 = 0
		for i in range(len(train_labels)):
			if train_labels[i][0] == 1:
				c0 = c0 + 1
			else:
				c1 = c1 + 1
		print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

		print ('Balancing training data...')
		min_c = min(c0, c1)
		idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
		idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
		new_indices = idx0[0:min_c] + idx1[0:min_c]
		print (len(new_indices))
		print (train_data.shape)
		train_data = train_data[new_indices,:,:,:]
		train_labels = train_labels[new_indices]


		train_size = train_labels.shape[0]

		c0 = 0
		c1 = 0
		for i in range(len(train_labels)):
			if train_labels[i][0] == 1:
				c0 = c0 + 1
			else:
				c1 = c1 + 1
		print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

		return [train_data, train_labels, train_size]

	# Make an image summary for 4d tensor image with index idx
	def get_image_summary(self, img, idx = 0):
		V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
		img_w = img.get_shape().as_list()[1]
		img_h = img.get_shape().as_list()[2]
		min_value = tf.reduce_min(V)
		V = V - min_value
		max_value = tf.reduce_max(V)
		V = V / (max_value*self.PIXEL_DEPTH)
		V = tf.reshape(V, (img_w, img_h, 1))
		V = tf.transpose(V, (2, 0, 1))
		V = tf.reshape(V, (-1, img_w, img_h, 1))
		return V
	
	# Make an image summary for 3d tensor image with index idx
	def get_image_summary_3d(self, img):
		V = tf.slice(img, (0, 0, 0), (1, -1, -1))
		img_w = img.get_shape().as_list()[1]
		img_h = img.get_shape().as_list()[2]
		V = tf.reshape(V, (img_w, img_h, 1))
		V = tf.transpose(V, (2, 0, 1))
		V = tf.reshape(V, (-1, img_w, img_h, 1))
		return V

	# Get prediction for given input image 
	def get_prediction(self, img):
		data = numpy.asarray(DU.img_crop(img, self.IMG_PATCH_SIZE, self.IMG_PATCH_SIZE))
		data_node = tf.constant(data)
		output = tf.nn.softmax(self.model(data_node))
		output_prediction = self.session.run(output)
		img_prediction = DU.label_to_img(img.shape[0], img.shape[1], self.IMG_PATCH_SIZE, self.IMG_PATCH_SIZE, output_prediction)

		return img_prediction

	def get_prediction_masks(self, filename, image_idx):
		imageid = "satImage_%.3d" % image_idx
		image_filename = filename + imageid + ".png"
		img = mpimg.imread(image_filename)

		img_prediction = self.get_prediction(img)

		return img_prediction

	# Get a concatenation of the prediction and groundtruth for given input file
	def get_prediction_with_groundtruth(self, filename, image_idx):

		imageid = "satImage_%.3d" % image_idx
		image_filename = filename + imageid + ".png"
		img = mpimg.imread(image_filename)

		img_prediction = self.get_prediction(img)
		cimg = DU.concatenate_images(img, img_prediction)

		return cimg

	# Get prediction overlaid on the original image for given input file
	def get_prediction_with_overlay(self, filename, image_idx):

		imageid = "satImage_%.3d" % image_idx
		image_filename = filename + imageid + ".png"
		img = mpimg.imread(image_filename)

		img_prediction = self.get_prediction(img)
		oimg = DU.make_img_overlay(img, img_prediction)

		return oimg
