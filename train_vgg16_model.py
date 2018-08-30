# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:46:14 2018

@author: n.muthuraj
"""


import numpy as np
from keras.models import Sequential
from keras.layers import Dense,   Conv2D, MaxPooling2D, Flatten, Dropout 
import create_dataset as cd
import cv2
from keras import optimizers
from matplotlib import pyplot
from keras.layers.convolutional import ZeroPadding2D
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.layers.normalization import BatchNormalization


train_data=cd.train_list_images
train_label=cd.train_labels
test_data=cd.test_list_images
test_label=cd.test_labels
val_data=cd.val_list_images
val_label=cd.val_labels

# build datset
x_train=np.zeros((len(train_data),224,224,3))
y_train=np.zeros((len(train_data),2))
for index in range(len(train_data)):
    x_train[index,:,:,:]=cv2.imread(train_data[index])
    y_train[index]=train_label[index]
#x_train=x_train.transpose(0,3,1,2)
#x_train=x_train.astype(np.int64)
#y_train=y_train.astype(np.int64)
y_train_age=y_train[:,0]
y_train_gender=y_train[:,1].astype(int)

x_test=np.zeros((len(test_data),224,224,3))
y_test=np.zeros((len(test_data),2))

for index in range(len(test_data)):
    x_test[index,:,:,:]=cv2.imread(test_data[index])
    y_test[index]=test_label[index]
#x_test=x_test.transpose(0,3,1,2)
#x_test=x_test.astype(np.int64)
#y_test=y_test.astype(np.int64)

y_test_age=y_test[:,0]
y_test_gender=y_test[:,1].astype(int)

x_val=np.zeros((len(val_data),224,224,3))
y_val=np.zeros((len(val_data),2))
for index in range(len(val_data)):
    x_val[index,:,:,:]=cv2.imread(val_data[index])
    y_val[index]=val_label[index]
#x_val=x_val.transpose(0,3,1,2)
#x_val=x_val.astype(np.int64)   
#y_val=y_val.astype(np.int64)


y_val_age=y_val[:,0]
y_val_gender=y_val[:,1].astype(int)
#VGG-16_architecture
    
del(y_train,y_test,y_val,index,test_data,test_label,train_data,train_label,val_data,val_label)

def VGG_16(weights_path=None, heatmap=False):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224,3)))
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu', name='conv3_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv4_4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_4'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(500, name='dense_3'))
    model.add(BatchNormalization())
    model.add(Dense(250, name='dense_4'))
    model.add(BatchNormalization())
    model.add(Dense(100, name='dense_5'))
    model.add(Dense(1, name='dense_6'))   
    
    return model


model= VGG_16()

adam = optimizers.adam(lr=0.001, beta_1 =  0.9, beta_2 = 0.999, epsilon = None, decay= 0, amsgrad = False)
#momentum = optimizers.SGD(lr=0.01, momentum = 0.9, decay=1e-6)
rms=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer = adam , loss = 'mean_absolute_error',metrics=['mse', 'mae', 'mape', 'cosine','accuracy'])


history=model.fit(x_train, y_train_age, batch_size = 10 , epochs = 25, validation_data=(x_val, y_val_age))
score = model.evaluate(x_test, y_test_age, batch_size = 10)
pyplot.plot(history.history['mean_squared_error'])
pyplot.show()
pyplot.plot(history.history['mean_absolute_error'])
pyplot.show()
pyplot.plot(history.history['mean_absolute_percentage_error'])
pyplot.show()
pyplot.plot(history.history['cosine_proximity'])
pyplot.show()

error=[]
for i in range(len(x_test)):
    error.append(int(model.predict(x_test[i].reshape(1,224,224,3))-y_test_age[i]))

model.save('ashik.h5')

#for i in range(len(test_data)):
#    if (train_label[i] > 90 or train_label[i] < 5):
#        print(train_data[i])
