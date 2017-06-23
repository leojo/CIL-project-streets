# ================================================================================================= #
#																									#
#	RPCA Algorithm using ADMM																		#
#																									#
#	RPCA finds the representation L + S = X where L is a low-rank matrix and S are the errors. 		#
#	We do this by minimizing ||L||* + u||S||1 undir the previous constraint.						#
#																									#
#	The idea is to find the L matrix for a street view image and feed that image to the CNN. 		#
#																									#
# ================================================================================================= #


import gzip
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

import code

import tensorflow.python.platform

import numpy
import tensorflow as tf

NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 20
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

tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist',
						   """Directory where to write event logs """
						   """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

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
	data = mpimg.imread(filename)
	return data[:,:,0].reshape((data.shape[0],data.shape[1]))

def bw_img_to_rgb(img):
	img_rgb = numpy.zeros((img.shape[0],img.shape[1],3), dtype=numpy.uint8)
	img_u8 = img_float_to_uint8(img)
	img_rgb[:,:,0] = img_u8
	img_rgb[:,:,1] = img_u8
	img_rgb[:,:,2] = img_u8
	return img_rgb


def concatenate_images(img, gt_img):
	nChannels = len(gt_img.shape)
	w = gt_img.shape[0]
	h = gt_img.shape[1]
	if nChannels == 3:
		cimg = numpy.concatenate((img, gt_img), axis=1)
	else:
		gt_img_3c = bw_img_to_rgb(gt_img)
		img8 = bw_img_to_rgb(img)
		cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
	return cimg


# applies s(x,t) on each element
def fS(X, t):
	f = lambda x : tf.multiply(tf.sign(x), tf.maximum(tf.subtract(tf.norm(x), t), 0))
	return tf.map_fn(f,X)



# D_t (X) = U * S_t(SIGMA) * V[trans]
def fD(X, t):
	s, u, v = tf.svd(X)
	s = tf.diag(s)
	print("shape u: %s, s: %s, v: %s"%(str(u.shape), str(s.shape), str(v.shape)))
	return tf.matmul(u, tf.matmul( fS(s, t), tf.transpose(v) ))


def nucNorm(M):
	return tf.trace(M) # this only works for square matriceS?

def oneNorm(M):
	return tf.norm(M, ord=1)

def condFunc(L, S, l, previous, current, epsilon):
	val = tf.reduce_mean(tf.abs(tf.subtract(current,previous)))
	print("The val is %s"%str(val))
	print("Is this boolean or what %s"%str((val < epsilon)))

	boolRes = tf.less(val,epsilon)
	return tf.Print(boolRes,[boolRes],message="INFORMATION:")

def bodyFunc(X, L, S, l, previous, current, mu, p, q, i):
	print("glubs")
	nextL = fD( X - S - tf.scalar_mul(q, l) , q)						# body 
	nextS = fS(X - nextL - tf.multiply(q, l), tf.multiply(mu, q))
	nextl = l + tf.scalar_mul(p, tf.subtract(tf.add(nextL, nextS), X) ) 

	return [nextL, nextS, nextl, current, tf.add(tf.trace(L), tf.scalar_mul(mu, tf.norm(S, ord=1))), i+1]


# ADMM = Alternating Direction Method of Multipliers.
# Builds on Dual Decomposition as Well as Method of Multipliers.
def rpca(X, mu, p): 
	# Recall: We are minimizing |L|* + mu*|S|1 w.r.t. L + S = X


	'''
		S = tf.zeros(shape=shape)
		L = X
		l = tf.zeros(shape=shape)
		t = 0
		q = 1.0/p

		nextVal = tf.constant(0, tf.float32) 	# Keep this 0 for now until we calculate the next value.
		currentVal = nucNorm(L) + mu*oneNorm(S) # This is our obj function L_* + mu * S_1

		currentVal = tf.add(tf.trace(L), tf.scalar_mul(mu, tf.norm(S, ord=1)))
		tf.while_loop(
			tf.assert_negative(tf.subtract(tf.reduce_mean(tf.abs(tf.subtract(nextVal,currentVal))), epsilon)), # cond


			nextL = fD(( X - S - tf.multiply(q, l) , q))							# body 
			nextS = fS(( X - nextL - tf.multiply(q, l), tf.multiply(mu, q)))
			nextl = l + tf.scalar_mul(p, tf.subtract(tf.add(nextL, nextS), X) ) 
			
			# update
			t = t+1
			L = nextL
			S = nextS
			l = nextL
			print("Working.... t = ")
			print(t)


			return {"Lt": L, "St": S, "t": t}	
		)
	'''
	epsilon = tf.constant(0.04, tf.float32)
	q = 1.0/p
	shape = tf.shape(X)
	print("X SHAPE %s"%str(X.shape))
	L0 = X
	S0 = tf.zeros(shape=shape)
	l0 = tf.zeros(shape=shape)
	i0 = tf.constant(0)
	previous0 = tf.constant(0, tf.float32)
	current0 = tf.add(tf.trace(L0), tf.scalar_mul(mu, tf.norm(S0, ord=1)))
	cond = lambda L, S, l, previous, current, i: i < 10 
	#cond = lambda L, S, l, previous, current: condFunc(L, S, l, previous, current, epsilon)
	#body = lambda L, S, l, previous, current: [fD( X - S - tf.multiply(q, l), q),  fS( X - nextL - tf.multiply(q, l), tf.multiply(mu, q)) , l + tf.scalar_mul(p, tf.subtract(tf.add(nextL, nextS), X) ), current, tf.add(tf.trace(L), tf.scalar_mul(mu, tf.norm(S, ord=1)))]
	body = lambda L, S, l, previous, current, i: bodyFunc(X, L, S, l, previous, current, mu, p, q, i)
	
	return tf.while_loop(cond, body, loop_vars = [L0, S0, l0, previous0, current0, i0])[0]

def converged(currentVal, nextVal, epsilon):
	diff = tf.subtract(nextVal,currentVal)
	return tf.assert_negative(tf.subtract(tf.reduce_mean(tf.abs(diff)), epsilon))

def main(argv=None):  # pylint: disable=unused-argument
	with tf.Session() as s:
		data_dir = 'output-masks/mask_10.png'

		# Extract it into numpy arrays.
		X = extract_data(data_dir, TRAINING_SIZE)
		print("0X SHAPE %s"%str(X.shape))
		res = rpca(X, 0.01, 0.01)


		imgBefore = X
		imgAfter = res.eval()

		print imgBefore.shape
		print imgAfter.shape

		cimg = concatenate_images(imgBefore, imgAfter)
		#fig1 = plt.figure(figsize=(10, 10))
		plt.imshow(cimg)
		#plt.imshow(imgs[imgIndx], cmap='Greys_r')
		plt.show()

if __name__ == '__main__':
	tf.app.run()


##################### ALGO OUTLINE FROM SLIDES
# Require X, mu, p > 0
# S_0 = 0, l_0 = 0, t = 0
#
# while not converged do
#	L_t+1 		:= 	D_{p-1} (X - S_t - {p-1}mat(l_t))
#	S_t+1 		:= 	S_mu_{p-1}(X-L_t+1 - {p-1}mat(l_t))
#	mat(l_t+1) 	:= 	mat(l_t) + p(L_t+1 + S_t+1 - X)
#	t = t + 1
# end while
# return L_t, S_t

# Some guidelines
#
# tf.norm: norm(tensor, ord='1') is the 1-norm.
#
# for basic math operations we have tf.add, tf.subtract, tf.multiply and tf.scalar_mul
#
# For example:
#				scalar_mul(scalar, x)
#				multiply(x, y) 
#				add(x, y) 
#
#	Actually we should use tf.matmul for matrix multiplication.
# etc.