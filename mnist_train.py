
from keras.datasets import mnist

import matplotlib.pyplot as plt
import h5py
from os import environ
from keras.utils import to_categorical

from keras import models,layers

from keras import optimizers

environ['TF_CPP_MIN_LOG_LEVEL']='3'

#Loading dataset
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

#print("Train Images shape: ", train_images.shape)
#print("Train LABELS shape: ", train_labels.shape)
#print("Test Images shape: ", test_images.shape)
#print("Test LABELS shape: ", test_labels.shape)


#print("image's label: ",train_labels[0])
#plt.imshow(train_images[0],cmap='Greys')

#Training,validation and test data set
valid_set = train_images[50000:60000]
train_set = train_images[0:50000]


#print(train_set.shape)


#creating tensors - tensors are N-dimensional Matrix (N > 2)
train_set = train_set.reshape((50000,28,28,1))
train_set = train_set.astype('float32') / 255 #Converting zero to one scale


valid_set = valid_set.reshape((10000,28,28,1))
valid_set = valid_set.astype('float32') / 255 #Converting zero to one scale


test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32') / 255 #Converting zero to one scale


valid_labels = train_labels[50000:60000]
train_labels = train_labels[0:50000]


#One-hot encoding
#print("Before One hot: ", train_labels[10])
train_labels = to_categorical(train_labels)
valid_labels = to_categorical(valid_labels)
test_labels = to_categorical(test_labels)
#print("After One hot: " ,train_labels[10])


#CNN Architecture
my_model = models.Sequential()

my_model.add(layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),use_bias=True,input_shape=(28,28,1)))
my_model.add(layers.Activation('relu'))

my_model.add(layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),use_bias=True))
my_model.add(layers.Activation('relu'))

my_model.add(layers.MaxPooling2D(pool_size=(2,2)))
my_model.add(layers.Dropout(rate=0.2))


my_model.add(layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),use_bias=True))
my_model.add(layers.Activation('relu'))

my_model.add(layers.MaxPooling2D(pool_size=(2,2)))

my_model.add(layers.Flatten())
my_model.add(layers.Dropout(rate=0.2))

my_model.add(layers.Dense(units=10,use_bias=True))
my_model.add(layers.Activation('relu'))

my_model.add(layers.Dense(units=10,use_bias=True))
my_model.add(layers.Activation('softmax'))


#CNN Summary
my_model.summary()

#CNN loss and optimizer
compilee =my_model.compile(optimizers.sgd(lr=0.1,decay=0.01),loss="categorical_crossentropy",metrics=['accuracy'])


#CNN training
my_model.fit(x=train_set,y=train_labels,batch_size=2500,epochs=20,validation_data=(valid_set,valid_labels))


#Save Model
my_model.save(filepath=r'C:\Users\user\Desktop\CNNudemy\handnumbers\model_save.h5',overwrite=True)

