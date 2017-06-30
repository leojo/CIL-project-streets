# ========================================================================================== #
#                                                                                            #
#                                                                                            #
#                                                                                            #
# ========================================================================================== #

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import code
import tensorflow.python.platform
import numpy as np
import tensorflow as tf
import datautils as DU

# Basically baseline program split to trainModel function and makePredictions function.
class CNN:

    NUM_CHANNELS = 3 # RGB images
    PIXEL_DEPTH = 255
    NUM_LABELS = 2
    TRAINING_SIZE = 100
    TEST_SIZE = 50
    VALIDATION_SIZE = 5  # Size of the validation set.
    SEED = 66478  # Set to None for random seed.
    BATCH_SIZE = 16 # 64
    NUM_EPOCHS = 10
    RECORDING_STEP = 1000
    # Set image patch size
    # IMG_PATCH_SIZE should be a multiple of 4
    # image size should be an integer multiple of this number!
    IMG_PATCH_SIZE = 400
    IMG_SAMPLE_SIZE = 128
    TRAIN_SAMPLES = 16
    
    def __init__(self, sess, flags):
        self.session = sess
        self.FLAGS = flags
        
        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when when we call:
        # {tf.initialize_all_variables().run()}
        self.conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, self.NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                                stddev=0.1,
                                seed=self.SEED))
        self.conv1_biases = tf.Variable(tf.zeros([32]))
        self.conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64],
                                stddev=0.1,
                                seed=self.SEED))
        self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))


        self.upsample2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 64, 64],  # 5x5 filter, depth 32.
                                stddev=0.1,
                                seed=self.SEED))
        self.deconv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 64, 32],  # 5x5 filter, depth 32.
                                stddev=0.1,
                                seed=self.SEED))
        self.deconv2_biases = tf.Variable(tf.zeros([32]))

        self.upsample1_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 32],  # 5x5 filter, depth 32.
                                stddev=0.1,
                                seed=self.SEED))
        self.deconv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, self.NUM_CHANNELS],
                                stddev=0.1,
                                seed=self.SEED))
        self.deconv1_biases = tf.Variable(tf.constant(0.1, shape=[self.NUM_CHANNELS]))

        self.output_weights = tf.Variable(
            tf.truncated_normal([5, 5, self.NUM_CHANNELS, self.NUM_LABELS],
                                stddev=0.1,
                                seed=self.SEED))
        self.output_biases = tf.Variable(tf.constant(0.1, shape=[self.NUM_LABELS]))

        

        self.saver = tf.train.Saver()

    def conv2d(self, inputs, weights, biases, filter_size=5, relu=True, stride=1):
        channels_in = inputs.get_shape().as_list()[-1]
        conv = tf.nn.conv2d(inputs,
                            weights,
                            strides=[1, stride, stride, 1],
                            padding='SAME')
        bias = tf.nn.bias_add(conv,biases)
        if relu:
            return tf.nn.relu(bias)
        return bias

    def max_pool2d(self,inputs):
        return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def upsample(self,inputs,weights,output_shape):
        return tf.nn.conv2d_transpose(inputs,filter=weights,output_shape=output_shape,strides=[1,2,2,1],padding="SAME")


    def makePredictions(self, RESTORE_MODEL=False):
        predMasks = []
        print ("Running prediction on test set")
        predictions_test_dir = "predictions_test/"
        test_data_filename = '../data/test_set_images/test_'
        if not os.path.isdir(predictions_test_dir):
            os.mkdir(predictions_test_dir)
        for i in range(1, self.TEST_SIZE+1):
            full_filename = test_data_filename+str(i)+'/'
            pimg = (self.get_prediction_masks(full_filename, i)*255).astype(np.uint8)
            Image.fromarray(pimg).save(predictions_test_dir + "mask_" + str(i) + ".png")     
            predMasks.append(pimg)

        return predMasks

    def makeTrainPredictions(self):
        train_data_filename = '../data/training/images/'
        print ("Running prediction on training set")
        prediction_training_dir = "predictions_training/"
        phase_2_training_dir = "training_phase2/"
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)
        for i in range(1, 101):
            pimg = self.get_prediction_with_groundtruth(train_data_filename, i)
            Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
            oimg = self.get_prediction_with_overlay(train_data_filename, i)
            oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")


            mimg = (self.get_training_prediction_masks(train_data_filename, i)*255).astype(np.uint8)
            Image.fromarray(mimg).save(phase_2_training_dir+"mask_"+str(i)+".png")

    def loadModel():
        try:
            print("Restoring model from "+self.FLAGS.train_dir + "/model.ckpt")
            self.saver.restore(self.session, self.FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")
        except Exception, e:
            raise Exception("No model was found! You need to train a model before making predictions.")   

    def trainModel(self, train_data_original, train_labels_original):
        train_data, train_labels, train_size = self.balanceTrainData(train_data_original, train_labels_original)

        #print("train_data.shape=",train_data.shape)
        #print("train_labels.shape=",train_labels.shape)

        train_data_node = tf.placeholder( tf.float32, 
                                            shape=(self.BATCH_SIZE, self.IMG_SAMPLE_SIZE, self.IMG_SAMPLE_SIZE, self.NUM_CHANNELS))
        train_labels_node = tf.placeholder(tf.float32,
                                            shape=(self.BATCH_SIZE, self.IMG_SAMPLE_SIZE, self.IMG_SAMPLE_SIZE, self.NUM_LABELS))

        logits = self.model(train_data_node)
        #print("train_data_node.shape=",train_data_node.get_shape().as_list())
        #print("train_labels_node.shape=",train_labels_node.get_shape().as_list())
        #print("logits.shape=",logits.get_shape().as_list())
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_labels_node)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

        # Run all the initializers to prepare the trainable parameters.

        tf.global_variables_initializer().run(session=self.session)
        print ('Initialized!')
        # Loop through training steps.
        print ('Total number of iterations = ' + str(int(self.NUM_EPOCHS * train_size / self.BATCH_SIZE)))

        training_indices = range(train_size)

        for iepoch in range(self.NUM_EPOCHS):

            # Permute training indices
            perm_indices = np.random.permutation(training_indices)
            print("Epoch %d of %d"%(iepoch+1,self.NUM_EPOCHS))

            for step in range (int(train_size / self.BATCH_SIZE)):

                offset = (step * self.BATCH_SIZE) % (train_size - self.BATCH_SIZE)
                batch_indices = perm_indices[offset:(offset + self.BATCH_SIZE)]

                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                batch_data = train_data[batch_indices, :, :, :]
                batch_labels = train_labels[batch_indices, :, :, :]
                print("Minibatch %d of %d:"%(step+1,int(train_size / self.BATCH_SIZE))),
                # This dictionary maps the batch data (as a np array) to the
                # node in the graph is should be fed to.
                feed_dict = {train_data_node: batch_data,
                             train_labels_node: batch_labels}
                _ , batch_loss = self.session.run([optimizer,loss],feed_dict=feed_dict)
                print("\t[Loss = %f]"%np.mean(batch_loss))


            # Save the variables to disk.
            save_path = self.saver.save(self.session, self.FLAGS.train_dir + "/model.ckpt")
            print("Model saved in file: %s" % save_path)

        self.makeTrainPredictions()

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(self, data):
        #Encoder
        conv1 = self.conv2d(data,
            weights=self.conv1_weights,
            biases =self.conv1_biases)
        pool1 = self.max_pool2d(conv1)

        conv2 = self.conv2d(pool1,
            weights=self.conv2_weights,
            biases =self.conv2_biases)
        pool2 = self.max_pool2d(conv2)

        #Decoder
        unpool2 = self.upsample(pool2,
            weights=self.upsample2_weights,
            output_shape=tf.shape(conv2))
        deconv2 = self.conv2d(unpool2,
            weights=self.deconv2_weights,
            biases =self.deconv2_biases,
            relu=False)

        unpool1 = self.upsample(deconv2,
            weights=self.upsample1_weights,
            output_shape=tf.shape(conv1))
        deconv1 = self.conv2d(unpool1,
            weights=self.deconv1_weights,
            biases =self.deconv1_biases,
            relu=False)
        out = self.conv2d(deconv1,
            weights=self.output_weights,
            biases =self.output_biases)

        return out

    def balanceTrainData(self, imgs, labs):
        print("Sampling balanced data")
        img_samples = []
        lab_samples = []
        num_per_ratio = [0]*10
        n,w,h,_ = imgs.shape
        for i in range(self.TRAIN_SAMPLES*10):
            m = np.random.randint(n)
            x = np.random.randint(w-self.IMG_SAMPLE_SIZE)
            y = np.random.randint(h-self.IMG_SAMPLE_SIZE)
            ratio = np.sum(labs[m,x:x+self.IMG_SAMPLE_SIZE,y:y+self.IMG_SAMPLE_SIZE,1]).astype(np.float32)/(self.IMG_SAMPLE_SIZE*self.IMG_SAMPLE_SIZE)
            ratio_cat = min(int(ratio*10),9)
            while num_per_ratio[ratio_cat]>=self.TRAIN_SAMPLES:
                m = np.random.randint(n)
                x = np.random.randint(w-self.IMG_SAMPLE_SIZE)
                y = np.random.randint(h-self.IMG_SAMPLE_SIZE)
                ratio = np.sum(labs[m,x:x+self.IMG_SAMPLE_SIZE,y:y+self.IMG_SAMPLE_SIZE,1]).astype(np.float32)/(self.IMG_SAMPLE_SIZE*self.IMG_SAMPLE_SIZE)
                ratio_cat = min(int(ratio*10),9)
            num_per_ratio[ratio_cat] += 1
            img_samples.append(imgs[m,x:x+self.IMG_SAMPLE_SIZE,y:y+self.IMG_SAMPLE_SIZE,:])
            lab_samples.append(labs[m,x:x+self.IMG_SAMPLE_SIZE,y:y+self.IMG_SAMPLE_SIZE,:])
        print("Done!")
        return np.asarray(img_samples),np.asarray(lab_samples),len(img_samples)


    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(self, img, idx = 0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value*self.PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V
    
    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(self, img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image 
    def get_prediction(self, img):
        data = np.asarray([img])
        data_node = tf.constant(data)
        output = tf.argmax(tf.nn.softmax(self.model(data_node)))
        output_prediction = self.session.run(output)

        return output_prediction

    def get_training_prediction_masks(self, filename, image_idx):
        imageid = "satImage_%.3d" % image_idx
        image_filename = filename+ imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = self.get_prediction(img)

        return img_prediction

    def get_prediction_masks(self, filename, image_idx):
        imageid = 'test_%i' % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = self.get_prediction(img)

        return img_prediction

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(self, filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename+ imageid + ".png"
        img = mpimg.imread(image_filename)
        img_prediction = self.get_prediction(img)
        cimg = DU.concatenate_images(img, img_prediction)

        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(self, filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = self.get_prediction(img)
        oimg = DU.make_img_overlay(img, img_prediction)

        return oimg
