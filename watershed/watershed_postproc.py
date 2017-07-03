import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

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
for i in range(1,51):
	img = cv2.imread("test_%d_mask.png"%i)
	thresh = img[:,:,0]

	# noise removal
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

	# sure background area
	sure_bg = cv2.dilate(opening,kernel,iterations=10)
	#show(img)
	#show(sure_bg)
	# Finding sure foreground area
	sure_fg = cv2.erode(opening,kernel,iterations=5)

	# Finding unknown region
	#show(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)

	markers = ((sure_fg/255)+1).astype(np.int32)
	markers[unknown==255] = 0

	#show(markers*127)
	#print(np.min(img),np.max(img),img.dtype,img.shape)
	#print(np.min(markers),np.max(markers),markers.dtype,markers.shape)
	#exit()

	markers = cv2.watershed(img,markers)
	img[markers == -1] = [0,0,0]
	img[markers == 1] = [0,0,0]
	img[markers == 2] = [255,255,255]

	save(img,"test_%d_mask_after.png"%i)
