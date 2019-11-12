from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys
from keras.models import load_model
import imutils
# import the necessary packages

from keras.preprocessing.image import img_to_array
import numpy as np
import argparse

image_size = 32
CLASSES_NUM = 62
init_lr = 1e-3
bs = 32
EPOCHS = 35

class Triffic_Net:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)

        model.add(Conv2D(20, (5, 5), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model

def load_data(path):
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(path)))
    print(imagePaths)
    random.seed(42)
    random.shuffle(imagePaths)

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)

        #图像大小32*32
        image = cv2.resize(image, (image_size, image_size))
        #转成keras类型
        image = img_to_array(image)
        data.append(image)

        label = int(imagePath.split(os.path.sep)[-2])
        labels.append(label)
    #归一化
    data = np.array(data, dtype='float')/255.0
    labels = np.array(labels)
    #传入数据到keras
    labels = to_categorical(labels, num_classes=CLASSES_NUM)

    return data,labels

def train(aug, trainX, trainY, testX, testY):
    print("[INFO] compiling model...")
    model = Triffic_Net.build(width=image_size, height=image_size, depth=3, classes=CLASSES_NUM)
    opt = Adam(lr=init_lr, decay=init_lr/EPOCHS)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=bs),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX)//bs,epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    mp = "iris_model.h5"
    model.save(mp)
    model.save('class.model')

    plt.style.use('ggplot')
    plt.figure()
    N = EPOCHS
    
    plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")


def predict():
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model("iris_model.h5")

    # load the image
    image = cv2.imread("2.png")
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image
    result = model.predict(image)[0]
    # print (result.shape)
    proba = np.max(result)
    label = str(np.where(result == proba)[0])
    label = "{}: {:.2f}%".format(label, proba * 100)
    print(label)

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)

if __name__ == '__main__':

    trainpath = 'C:/Users/admin/Desktop/photoclass/S2/traffic-sign/traffic-sign/train'
    testpath = 'C:/Users/admin/Desktop/photoclass/S2/traffic-sign/traffic-sign/test'

    trainx, trainy = load_data(trainpath)
    testx, testy = load_data(testpath)

    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    
    train(aug, trainx, trainy, testx, testy)


    #predict()

