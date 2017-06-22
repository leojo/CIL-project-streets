# Info

I am trying to train a network to classify each pixel.
* generate for each pixel a patch
* train with this patch and the given label the network

Test:
not yet done => but each pixel can be classified by sending the patch through the network. The pictures and patches are already loaded (limited to only load some pictures, one or two currently), but the classification for the testing images has to be implemented after the training.

Tutorial: http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/