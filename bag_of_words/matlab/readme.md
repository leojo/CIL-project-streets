# Info

Please only take a look in the kmeans_evalutaion.m. Some functions which I had to overwrite from my previous assignment. Also redefine the paths to the training and testing images.. (line 18 and 19)

There is one drawback: It converts the image into gray-scale, which is not good at all. It does this because the descriptors are represented as gradients. I will try to change this to take the color-histogram or something like this.

Additionally, for testing purpose I only take each x-th pixel (where x is the parameter stride) for testing and training. This is of course also something which must be changed for later use.

Because we are not interested in "is there a street on the image" but "where is the street in the image", the bag-of-words is probably not really the best option. Currently it is not a real bag-of-words anymore, but simply checking each pixel if the closest "word" is learned as a street or as something else.

The words are summarized by a k-means. This could be changed maybe to something even better?! PCA or whatever to learn about streets or not streets.

Please let me know if you have some questions. This evening I try to figure out the convolutional-network with python and then I can maybe after some inputs proceed with this approach here.