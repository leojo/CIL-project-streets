# ========================================================================================== #
# MAIN SCRIPT:																				 #
#	Intended to utilize the other classes to prepare the data, train the FCN on the data 	 #
#	and perform postprocessing. 															 #	
#																							 #
#	Alternatively, load up a pre_existing model and make predictions on test data.			 #
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
from simplefcn import SimpleFCN
import datautils as DU
import preprocessor as PREP
import postprocessor as POSTP

tf.app.flags.DEFINE_string('train_dir', 'model',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS

def strToBool(v):
  return v in ("T", "True", "t", "true")

def main(argv=None):  # pylint: disable=unused-argument
	
	# Create a local session to run this computation.
	with tf.Session() as s:

		fcn = SimpleFCN(s, FLAGS);

		# If this isn't done then fcn tries to load last model.
		train = strToBool(str(sys.argv[1]))
		if(train):
			data_dir = '../data/training/'
			train_data, gtruth_data = DU.prepareTrainData(data_dir, fcn.IMG_PATCH_SIZE)
			train_data_preproc = PREP.preprocess(train_data);
			# Train using our FCN
			fcn.trainModel(train_data_preproc, gtruth_data) 
		
		# Prepare and preprocess test data so we can make predictions:
		#test_data = DU.prepareTestData();
		#test_data_preproc = PREP.preprocess(test_data);

		output_masks = fcn.makePredictions()
		#output_masks_postproc = POSTP.postprocess(output_masks)

		#DU.generateImages(output_masks_postproc)
		#DU.generateSubmissionFile(output_masks_postproc)

	# =====================================================


if __name__ == '__main__':
    tf.app.run()