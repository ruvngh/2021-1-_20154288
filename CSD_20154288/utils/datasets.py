import os
import numpy as np
from tensorflow.keras.utils import Sequence
from imageio import imread
import cv2


def get_labels():
    return ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger',
            'Contempt']


class DataManager(object):

    def __init__(self, dataset_path=None, target_size=(64, 64), rescale=None):
        self.dataset_path = dataset_path
        self.target_size = target_size
        self.rescale = rescale

        self._load_affect_net()

    def _load_affect_net(self):
        self.train_image_path = os.path.join(self.dataset_path, 'train_set', 'images')
        self.train_label_path = os.path.join(self.dataset_path, 'train_set', 'annotations')

        self.val_image_path = os.path.join(self.dataset_path, 'val_set', 'images')
        self.val_label_path = os.path.join(self.dataset_path, 'val_set', 'annotations')

        self.train_images, self.train_labels = self._extract(self.train_image_path,
                                                             self.train_label_path)

        self.val_images, self.val_labels = self._extract(self.val_image_path,
                                                         self.val_label_path)

    def _extract(self, image_path, label_path):

        # check_label_exists.
        image_aggregation = label_path + "/image_path_total.npy"
        label_aggregation = label_path + "/label_total.npy"

        if os.path.exists(image_aggregation) and os.path.exists(label_aggregation):

            print('load data...')

            images = np.load(image_aggregation)
            labels = np.load(label_aggregation)

            return images, labels

        else:
            print('processing data...')

            # image_path.
            images = []

            # label data. [one_hot(exp, 8), val, aro] len : 10
            labels = []
            print('size : ', len(os.listdir(image_path)))
            for i in os.listdir(image_path):
                images.append(os.path.join(image_path, i))

                # exp one hot.
                temp = np.zeros(8)
                temp[int(np.load(label_path + '/' + i.split('.')[0] + '_exp.npy'))] = 1

                # add val.
                temp = np.append(temp, float(
                    np.load(label_path + '/' + i.split('.')[0] + '_val.npy')))

                # add aro.
                temp = np.append(temp, float(
                    np.load(label_path + '/' + i.split('.')[0] + '_aro.npy')))

                labels.append(temp)

            # save image path and label data.
            np.save(image_aggregation, np.array(images))
            np.save(label_aggregation, np.array(labels))

            return images, labels

    def get_train_dataset(self):
        # return path.
        return self.train_images, np.array(self.train_labels)[:, :8], \
               np.array(self.train_labels)[:, 8:]

    def get_val_dataset(self):
        # return datasets
        # append all images. and resize to target_size.
        image_vals = []
        for im in self.val_images:
            temp_img = imread(im)
            temp_img = cv2.resize(temp_img, self.target_size)
            image_vals.append(temp_img)

        image_processed = np.array(image_vals).astype(np.float)

        if self.rescale:
            image_processed *= self.rescale

        return image_processed, \
               (np.array(self.val_labels)[:, :8]).astype(int), \
               np.array(self.val_labels)[:, 8:]


class DataGenerator(Sequence):
    def __init__(self, imgs, exp, etc, batch_size, target_size=(64, 64), shuffle=True,
                 rescale=None):
        self.images, self.exp, self.etc = imgs, exp, etc
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.new_train_indices = np.arange(len(self.images))
        self.rescale = rescale

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.new_train_indices)

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):  # n th batch.
        # todo: image resize.
        batch_index = self.new_train_indices[
                      index * self.batch_size: (index + 1) * self.batch_size]

        data_image_list = self.images[batch_index]
        data_exp_list = self.exp[batch_index]
        data_etc_list = self.etc[batch_index]

        image_trains = []
        for im in data_image_list:
            temp_img = imread(im)
            temp_img = cv2.resize(temp_img, self.target_size)
            image_trains.append(temp_img)

        image_processed = np.array(image_trains).astype(np.float)

        if self.rescale:
            image_processed *= self.rescale

        return image_processed, {'predictions_exp': np.array(data_exp_list).astype(int),
                                 'predictions_etc': np.array(data_etc_list)}
