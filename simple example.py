# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:40:35 2020

@author: DODO
"""


import os,sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.preprocessing import image
#from tensorflow.python.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
#from keras.applications.vgg16 import VGG16
#from keras.preprocessing import image
#from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
import random

O_path='H:/清华大学/导师安排/20220218瑞华煤炭矿业协会奖申报/20220615代码/岩性识别/Rock Dataset/Field Scale/'

def percent(value):
    return '%.2f%%' % (value * 100)

def DataSet():
    
    train_path_1 = O_path+'Igneous Rock/'
    train_path_2 = O_path+'Metamorphic Rock/'
    train_path_3 = O_path+'Sedimentary Rock/'
    
    test_path_1 = O_path+'Igneous Rock/'
    test_path_2 = O_path+'Metamorphic Rock/'
    test_path_3 = O_path+'Sedimentary Rock/'
    
    imglist_train_1 = os.listdir(train_path_1)
    imglist_train_2 = os.listdir(train_path_2)
    imglist_train_3 = os.listdir(train_path_3)
    
    imglist_test_1 = os.listdir(test_path_1)
    imglist_test_2 = os.listdir(test_path_2)
    imglist_test_3 = os.listdir(test_path_3)
        
    X_train = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3), 224, 224, 3))
    Y_train = np.empty((len(imglist_train_1) + len(imglist_train_2) +len(imglist_train_3), 3))  
    
    count = 0
    
    for img_name in imglist_train_1:
        
        img_path = train_path_1 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((1,0,0))
        count+=1
        
    for img_name in imglist_train_2:

        img_path = train_path_2 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((0,1,0))
        count+=1
    
    for img_name in imglist_train_3:

        img_path = train_path_3 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((0,0,1))
        count+=1
        
    X_test = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3), 224, 224, 3))
    Y_test = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3), 3))
    count = 0
    for img_name in imglist_test_1:

        img_path = test_path_1 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((1,0,0))
        count+=1
        
    for img_name in imglist_test_2:
        
        img_path = test_path_2 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((0,1,0))
        count+=1
    for img_name in imglist_test_3:
        
        img_path = test_path_3 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((0,0,1))
        count+=1
        
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    
    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]    
    Y_test = Y_test[index]

    return X_train,Y_train,X_test,Y_test


X_train,Y_train,X_test,Y_test = DataSet()
print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)


# # model
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(3,activation='sigmoid'))

##model = ResNet50( weights=None,classes=3)
##model = VGG16(weights='imagenet', include_top=True)

##model.compile(optimizer=tf.train.AdamOptimizer(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

# # train


model.fit(X_train, Y_train, epochs=20, batch_size=4)

# # evaluate


model.evaluate(X_test, Y_test, batch_size=8)

# # save


model.save('my_rock_model.h5')

# # restore


model = tf.keras.models.load_model('my_rock_model.h5')

# # test



img_path = O_path+'Sedimentary Rock/'

img = image.load_img(img_path, target_size=(224, 224))

plt.imshow(img)
img = image.img_to_array(img)/ 255.0
img = np.expand_dims(img, axis=0)  # 为batch添加第四维

print(model.predict(img))
a = model.predict(img)[0]
b = a[0] 
c = a[1] 
d = a[2]
if b > c :
    if b > d : 
        print('这是变质岩')
    else:
        print('这是岩浆岩')
else:
    if c > d :
        print('这是沉积岩')
    else:
        print('这是岩浆岩')


    