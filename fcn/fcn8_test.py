import logging
import os
import sys

import scipy as scp
import scipy.misc
import tensorflow as tf
from matplotlib import pyplot as plt

from fcn import fcn8_vgg

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img1 = scp.misc.imread("./fcn/test_data/tabby_cat.png")
img1 = scp.misc.imread("./fcn/test_data/test_1.png")

with tf.Session() as sess:
    images = tf.placeholder("float")
    feed_dict = {images: img1}
    batch_images = tf.expand_dims(images, 0)

    vgg_fcn = fcn8_vgg.FCN8VGG()
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True, num_classes=2)

    print('Finished building Network.')

    logging.warning("Score weights are initialized random.")
    logging.warning("Do not expect meaningful results.")

    logging.info("Start Initializing Variabels.")

    init = tf.global_variables_initializer()
    sess.run(init)

    print('Running the Network')
    tensors = [vgg_fcn.pred, vgg_fcn.pred_up]
    down, up = sess.run(tensors, feed_dict=feed_dict)

    # plt.imshow(down[0])
    # plt.show()
    plt.imshow(up[0])
    plt.show()
    # down_color = utils.color_image(down[0])
    # up_color = utils.color_image(up[0])
    #
    # scp.misc.imsave('fcn8_downsampled.png', down_color)
    # scp.misc.imsave('fcn8_upsampled.png', up_color)