import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle
import matplotlib.image as mpimg


def load_train(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('Reading training images')
    # for fld in classes:  # assuming data directory has a separate folder for each class, and that each folder is named after the class
    #     index = classes.index(fld)
    #     print('Loading {} files (Index: {})'.format(fld, index))

    path = os.path.join(train_path, 'images', '*g')
    path_label = os.path.join(train_path, 'groundtruth', '*g')

    files = glob.glob(path)
    files_label = glob.glob(path_label)

    for i in range(1, min(len(files), 2)):
        # for fl in files:
        fl = files[i]
        fl_label = files_label[i]

        print('Loading {} files (Index: {})'.format(path, i))

        image = cv2.imread(fl)
        image_size = len(image)

        # image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
        image_label = mpimg.imread(fl_label)
        # image_label = cv2.imread(fl_label)

        for x in range(16, image_size-32):
            for y in range(16, image_size-32):
                patch = image[x - 16:x + 16, y-16:y+16]

                if image_label[x, y] > 0:
                    pixel_class = 1
                else:
                    pixel_class = 0

                images.append(patch)
                label = np.zeros(len(classes))
                label[pixel_class] = 1.0
                labels.append(label)
                ids.append(str(i) + "_" + str(x) + "_" + str(y))
                cls.append(pixel_class)

    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


def load_test(test_path, image_size, classes):
    path = os.path.join(test_path, '*')
    directories = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    print("Reading test images")
    for d in range(1, min(len(directories), 2)):
        # for dir in directories:
        dir = directories[d]
        img_path = os.path.join(dir, '*g')
        files = sorted(glob.glob(img_path))

        for fl in files:
            flbase = os.path.basename(fl)
            print(fl)
            image = cv2.imread(fl)
            image_size = len(image)
            # img = cv2.resize(img, (image_size, image_size), cv2.INTER_LINEAR)

            for x in range(16, image_size - 32):
                for y in range(16, image_size - 32):
                    patch = image[x - 16:x + 16, y - 16:y + 16]

                    X_test.append(patch)
                    X_test_id.append(str(fl) + "_" + str(x) + "_" + str(y))

    ### because we're not creating a DataSet object for the test images, normalization happens here
    X_test = np.array(X_test, dtype=np.uint8)
    X_test = X_test.astype('float32')
    X_test = X_test / 255

    return X_test, X_test_id


class DataSet(object):
    def __init__(self, images, labels, ids, cls):
        self._num_examples = images.shape[0]
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def ids(self):
        return self._ids

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # # Shuffle the data (maybe)
            # perm = np.arange(self._num_examples)
            # np.random.shuffle(perm)
            # self._images = self._images[perm]
            # self._labels = self._labels[perm]
            # Start next epoch

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size=0):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, ids, cls = load_train(train_path, image_size, classes)
    images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_ids = ids[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_ids = ids[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

    return data_sets


def read_test_set(test_path, image_size, classes):
    images, ids = load_test(test_path, image_size, classes)
    return images, ids
