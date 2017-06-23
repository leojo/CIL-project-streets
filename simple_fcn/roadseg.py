# ========================================================================================== #
# MAIN SCRIPT:																				 #
#	Intended to utilize the other classes to prepare the data, train the FCN on the data 	 #
#	and perform postprocessing. 															 #	
#																							 #
#	Alternatively, load up a pre_existing model and make predictions on test data.			 #
#																							 #
# ========================================================================================== #

import SimpleFCN as SFCN
import DataUtils as DU
from preprocessor import PREP
from postprocessor import POSTP


tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS



def main(argv=None):  # pylint: disable=unused-argument

	# IDEALLY:
	# =====================================================
	train = True
	data_dir = 'data/training/'

	if(train):
		train_data, gtruth_data = DU.prepareTrainData(data_dir);
		train_data_preproc = PREP.preprocess(train_data);
		# Train using our FCN
		SFCN.train(train_data_preproc) 
	

	# Prepare and preprocess test data so we can make predictions:
	test_data = DU.prepareTestData();
	test_data_preproc = PREP.preprocess(test_data);

	output_masks = SFCN.predict(test_data_preproc)
	output_masks_postproc = POSTP.postprocess(output_masks)

	DU.generateImages(output_masks_postproc)
	DU.generateSubmissionFile(output_masks_postproc)

	# =====================================================


if __name__ == '__main__':
    tf.app.run()