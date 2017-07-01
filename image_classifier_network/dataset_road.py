import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle
import matplotlib.image as mpimg


def load_train(train_path, image_size, classes):
    images = np.array([])
    labels = np.array([])
    ids = np.array([])
    cls = np.array([])

    print('Reading training images')

    path = os.path.join(train_path, 'images', '*g')
    path_label = os.path.join(train_path, 'groundtruth', '*g')

    files = glob.glob(path)
    files_label = glob.glob(path_label)

    for i in range(0, min(len(files), 100)):
        images_pos = []
        labels_pos = []
        ids_pos = []
        cls_pos = []

        images_neg = []
        labels_neg = []
        ids_neg = []
        cls_neg = []

        # for fl in files:
        fl = files[i]
        fl_label = files_label[i]

        print('Loading {} files (Index: {})'.format(path, i))

        image = cv2.imread(fl)
        orig_image_size = len(image)
        image = np.lib.pad(image, ((16,16),(16,16),(0,0)), 'edge')

        image_label = mpimg.imread(fl_label)
        cls_0 = 0
        cls_1 = 0

        for x in range(0, orig_image_size, 8):
            for y in range(0, orig_image_size, 8):
                patch = image[x:x+32, y:y+32]

                if image_label[x, y] > 0:
                    pixel_class = 1
                    cls_1 += 1

                    label = np.zeros(len(classes))
                    label[pixel_class] = 1.0

                    images_pos.append(patch)
                    labels_pos.append(label)
                    ids_pos.append(str(i) + "_" + str(x) + "_" + str(y))
                    cls_pos.append(pixel_class)
                else:
                    pixel_class = 0
                    cls_0 += 1

                    label = np.zeros(len(classes))
                    label[pixel_class] = 1.0

                    images_neg.append(patch)
                    labels_neg.append(label)
                    ids_neg.append(str(i) + "_" + str(x) + "_" + str(y))
                    cls_neg.append(pixel_class)

        print("Streets: " + str(cls_1) + ", Other: " + str(cls_0))

        min_samples = min(cls_1, cls_0)
        shuffle(images_neg, labels_neg, ids_neg, cls_neg)
        shuffle(images_pos, labels_pos, ids_pos, cls_pos)

        temp_images = []
        temp_labels = []
        temp_ids = []
        temp_cls = []

        for c in range(0, min_samples-1):
            temp_images.append(images_pos[c])
            temp_labels.append(labels_pos[c])
            temp_ids.append(ids_pos[c])
            temp_cls.append(cls_pos[c])

            temp_images.append(images_neg[c])
            temp_labels.append(labels_neg[c])
            temp_ids.append(ids_neg[c])
            temp_cls.append(cls_neg[c])

        if len(images) == 0:
            images = np.array(temp_images)
            labels = np.array(temp_labels)
            ids = np.array(temp_ids)
            cls = np.array(temp_cls)
        else:
            images = np.concatenate((images, temp_images), axis=0)
            labels = np.concatenate((labels, temp_labels), axis=0)
            ids = np.concatenate((ids, temp_ids), axis=0)
            cls = np.concatenate((cls, temp_cls), axis=0)

    print("length of samples: " + str(len(images)))

    return images, labels, ids, cls


def load_test_image(id, image_path):
    x_test = []
    x_test_id = []
    print("Reading test image: " + image_path)

    image = cv2.imread(image_path)
    orig_image_size = len(image)
    image = np.lib.pad(image, ((16,16),(16,16),(0,0)), 'edge')
    image = np.array(image, dtype=np.uint32)
    image = image.astype(np.float32)
    image = image / 255

    for x in range(0, orig_image_size):
        for y in range(0, orig_image_size):
            patch = image[x :x + 32, y:y + 32]

            x_test.append(patch)
            x_test_id.append(str(id) + "_" + str(x) + "_" + str(y))

    x_test = np.array(x_test)

    return x_test, x_test_id


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
            # self._ids = self._ids[perm]
            # self._cls = self._cls[perm]

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
    images, labels, ids, cls = shuffle(images, labels, ids, cls)

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


def read_test_set(id, test_path):
    images, ids = load_test_image(id, test_path)
    return images, ids
