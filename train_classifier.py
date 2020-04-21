from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import facenet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import math
import pickle
from sklearn.svm import SVC


with tf.Graph().as_default():

    with tf.Session() as sess:

        datadir = './images'
        dataset = facenet.get_dataset(datadir)
        paths, labels = facenet.get_image_paths_and_labels(dataset)

        print('Loading feature extraction model')
        modeldir = './model/20180402-114759.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        #print(images_placeholder)
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        #rint(embeddings)
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        #print(phase_train_placeholder)
        embedding_size = embeddings.get_shape()[1]
        #print(embedding_size)

        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        batch_size = 1000
        image_size = 160
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

        classifier_filename = './my_classifier.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        # Train classifier
        print('Training classifier')
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        model = SVC(kernel='linear', probability=True)
        model.fit(emb_array, labels)

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

    

