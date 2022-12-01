from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf


X = []
for filename in os.listdir('/Users/ximenagonzalez/Desktop/AM-PROY-2/Full-version/Train/'):
    try:
        X.append(img_to_array(load_img('/Users/ximenagonzalez/Desktop/AM-PROY-2/Full-version/Train/'+filename)))
    except:
        pass

X = np.array(X, dtype=float)

# Set up train and test data
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain

print(Xtrain.shape)

model = Sequential()
model.add(InputLayer(input_shape=(400, 400, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='rmsprop', loss='mse')

# model.summary()

datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

batch_size = 10


def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)


tensorboard = TensorBoard(log_dir="/Users/ximenagonzalez/Desktop/AM-PROY-2/output/first_run")
model.fit_generator(image_a_b_gen(batch_size), callbacks=[tensorboard], epochs=10, steps_per_epoch=950)

print("die 1")

model_json = model.to_json()
with open("/Users/ximenagonzalez/Desktop/AM-PROY-2/Beta-version/model2.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("/Users/ximenagonzalez/Desktop/AM-PROY-2/Beta-version/model2.h5")

Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
Ytest = Ytest / 128
# print(Xtest.shape)

#print(model.evaluate(Xtest, Ytest, batch_size=batch_size))
# print(model.evaluate(Xtest, Ytest, batch_size = 1))
print("die 2")

color_me = []
for filename in os.listdir('/Users/ximenagonzalez/Desktop/AM-PROY-2/Full-version/Test/'):
    color_me.append(img_to_array(load_img('/Users/ximenagonzalez/Desktop/AM-PROY-2/Full-version/Test/'+filename)))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

output = model.predict(color_me)
output = output * 128

print("die 3")

for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("/Users/ximenagonzalez/Desktop/AM-PROY-2/Beta-version/result/img_"+str(i)+".png", lab2rgb(cur))

print("DONE")

