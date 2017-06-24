# ========================================================================================== #
# POSTPROCESSOR:																	 	 #
#	For postprocessing functions to be used on the prediction outcomes of the FCN.			 #																		     
#																							 #
# ========================================================================================== #


# POSSIBLE POSTPROCESSING IDEAS
# Some kind of denoising: RPCA, Bag-of-Words approach, etc.

def postprocess(output_masks):
	return output_masks