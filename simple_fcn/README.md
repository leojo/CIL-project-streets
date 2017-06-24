This is my attempt at a simplified version of the FCN we have been looking at.

I define different classes for each step with the intention that we could easily substitute and try out different preprocessing or postprocessing steps, as well as experiment with the network independently. 

roadseg.py is the main script, it uses the other classes to prepare our data, preprocess it, run it through the training network which then saves the model. Afterwards it can use the model (or load a saved model) to make predictions, generate prediction masks and a submission file.

																		Andrea Bjornsdottir