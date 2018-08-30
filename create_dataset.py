# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:35:43 2018

@author: n.muthuraj
"""

from random import shuffle
import sys
import cv2
import tensorflow as tf
import clean_data 


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(addr,img)
    return img
 
def createDataRecord(out_filename, list_images, labels):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(list_images)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print('Train data: {}/{}'.format(i, len(list_images)))
            sys.stdout.flush()
        # Load the image
        img = load_image(list_images[i])

        label = labels[i]

        if img is None:
            continue

        # Create a feature
        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()


list_images=clean_data.list_images
labels=clean_data.labels

c = list(zip(list_images, labels))
shuffle(c)
list_images, labels=zip(*c)  
    
# Divide the data into 60% train, 20% validation, and 20% test
train_list_images = list_images[0:int(0.6*len(list_images))]
train_labels = labels[0:int(0.6*len(labels))]
val_list_images = list_images[int(0.6*len(list_images)):int(0.8*len(list_images))]
val_labels = labels[int(0.6*len(list_images)):int(0.8*len(list_images))]
test_list_images = list_images[int(0.8*len(list_images)):]
test_labels = labels[int(0.8*len(labels)):]



# Creation of TF records
#createDataRecord('train.tfrecords', train_list_images, train_labels)
#createDataRecord('val.tfrecords', val_list_images, val_labels)
#createDataRecord('test.tfrecords', test_list_images, test_labels)


# clean up un necessary variables

del (c,list_images,labels)