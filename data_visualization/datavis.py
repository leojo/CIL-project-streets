import gzip
import os
import sys
import urllib
import matplotlib as plt
import matplotlib.image as mpimg
from PIL import Image

from resizeimage import resizeimage
import matplotlib.pyplot as plt
import code

import tensorflow.python.platform

import numpy
import tensorflow as tf

# Some functions from baseline + other functions to see how we might change images to improve features.
# For example can reduce resolution somewhat, perhaps saturate images etc.

NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 50
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 5
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000

# Set image patch size
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

# Extract patches from a given image
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

def extract_data(filename, num_images):
	"""Extract the images into a 4D tensor [image index, y, x, channels].
	Values are rescaled from [0, 255] down to [-0.5, 0.5].
	"""
	imgs = []
	for i in range(1, num_images+1):
		imageid = "satImage_%.3d" % i
		image_filename = filename + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Loading ' + image_filename)
			img = mpimg.imread(image_filename)
			imgs.append(img)
		else:
			print ('File ' + image_filename + ' does not exist')

	num_images = len(imgs)
	IMG_WIDTH = imgs[0].shape[0]
	IMG_HEIGHT = imgs[0].shape[1]
	N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

	img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
	data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

	return numpy.asarray(data)

def extract_imgs(filename, num_images):
	"""Extract the images into a 4D tensor [image index, y, x, channels].
	Values are rescaled from [0, 255] down to [-0.5, 0.5]."""
	imgs = []
	for i in range(1, num_images+1):
		imageid = "satImage_%.3d" % i
		image_filename = filename + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Loading ' + image_filename)
			img = mpimg.imread(image_filename)
			imgs.append(img)
		else:
			print ('File ' + image_filename + ' does not exist')

	return imgs


		
# Assign a label to a patch v
def value_to_class(v):
	foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
	df = numpy.sum(v)
	if df > foreground_threshold:
		return [0, 1]
	else:
		return [1, 0]

# Extract label images
def extract_labels(filename, num_images):
	"""Extract the labels into a 1-hot matrix [image index, label index]."""
	gt_imgs = []
	for i in range(1, num_images+1):
		imageid = "satImage_%.3d" % i
		image_filename = filename + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Loading ' + image_filename)
			img = mpimg.imread(image_filename)
			gt_imgs.append(img)
		else:
			print ('File ' + image_filename + ' does not exist')

	num_images = len(gt_imgs)
	gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
	data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
	labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

	# Convert to dense 1-hot representation.
	return labels.astype(numpy.float32)

	# Extract label images
def extract_gt_imgs(filename, num_images):
	"""Extract the labels into a 1-hot matrix [image index, label index]."""
	gt_imgs = []
	for i in range(1, num_images+1):
		imageid = "satImage_%.3d" % i
		image_filename = filename + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Loading ' + image_filename)
			img = mpimg.imread(image_filename)
			gt_imgs.append(img)
		else:
			print ('File ' + image_filename + ' does not exist')

	# Convert to dense 1-hot representation.
	return gt_imgs


def error_rate(predictions, labels):
	"""Return the error rate based on dense predictions and 1-hot labels."""
	return 100.0 - (
		100.0 *
		numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
		predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
	max_labels = numpy.argmax(labels, 1)
	max_predictions = numpy.argmax(predictions, 1)
	file = open(filename, "w")
	n = predictions.shape[0]
	for i in range(0, n):
		file.write(max_labels(i) + ' ' + max_predictions(i))
	file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
	max_labels = numpy.argmax(labels, 1)
	max_predictions = numpy.argmax(predictions, 1)
	print (str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
	array_labels = numpy.zeros([imgwidth, imgheight])
	idx = 0
	for i in range(0,imgheight,h):
		for j in range(0,imgwidth,w):
			if labels[idx][0] > 0.5:
				l = 1
			else:
				l = 0
			array_labels[j:j+w, i:i+h] = l
			idx = idx + 1
	return array_labels

def img_float_to_uint8(img):
	rimg = img - numpy.min(img)
	rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
	return rimg

def concatenate_images(img, gt_img):
	nChannels = len(gt_img.shape)
	w = gt_img.shape[0]
	h = gt_img.shape[1]
	if nChannels == 3:
		cimg = numpy.concatenate((img, gt_img), axis=1)
	else:
		gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
		gt_img8 = img_float_to_uint8(gt_img)          
		gt_img_3c[:,:,0] = gt_img8
		gt_img_3c[:,:,1] = gt_img8
		gt_img_3c[:,:,2] = gt_img8
		img8 = img_float_to_uint8(img)
		cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
	return cimg

def make_img_overlay(img, predicted_img):
	w = img.shape[0]
	h = img.shape[1]
	color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
	color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

	img8 = img_float_to_uint8(img)
	background = Image.fromarray(img8, 'RGB').convert("RGBA")
	overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
	new_img = Image.blend(background, overlay, 0.2)
	return new_img


def modify_imgs(filename, num_images): 
	for i in range(1, num_images+1):
		imageid = "satImage_%.3d" % i
		image_filename = filename + imageid + ".png"
		new_image_filename = 'modifieddata/' + imageid + ".png"
		if os.path.isfile(image_filename):
			print ('Loading ' + image_filename)
			img = Image.open(image_filename)
			modImg = img.resize((img.size[0], img.size[1]), Image.BICUBIC)
			modImg.save(new_image_filename)
		else:
			print ('File ' + image_filename + ' does not exist')



def main(argv=None): 
	print('running....')

	data_dir = '../data/training/'
	train_data_filename = data_dir + 'images/'
	train_labels_filename = data_dir + 'groundtruth/' 
	# Extract it into numpy arrays.
	imgs = modify_imgs(train_data_filename, TRAINING_SIZE)
	#gt_imgs = extract_gt_imgs(train_labels_filename, TRAINING_SIZE)

	#imgIndx = 5
	#print('this is the current shape')
	#print(imgs[imgIndx].size)
	#tweekedImg = resizeimage.resize_cover(imgs[imgIndx], [100, 100])
	# Show first image and its groundtruth image
	#cimg = concatenate_images(tweekedImg, gt_imgs[imgIndx])
	#fig1 = plt.figure(figsize=(10, 10))
	#plt.imshow(imgs[imgIndx], cmap='Greys_r')
	#plt.show()



if __name__ == '__main__':
	main()