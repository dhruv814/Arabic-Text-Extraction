import numpy as np
import os
import cv2
from tqdm import tqdm
import pickle
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
import random
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten,Dropout,BatchNormalization,MaxPool2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

"""DATADIR = """""

CATEGORIES =["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15","c16","c17","c18","c19","c20","c21","c22","c23","c24","c25","c26"
              ,"c27","c28"]

"""training_data = []
IMG_SIZE = 50

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:

                pass
create_training_data()
    #print(training_data)
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(y)
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3)
print(len(X))"""
x_train = pd.read_csv("../CNN-2/csvTrainImages 13440x1024.csv",header=None)
y_train = pd.read_csv("../CNN-2/csvTrainLabel 13440x1.csv",header=None)

x_val = pd.read_csv("../CNN-2/csvTestImages 3360x1024.csv",header=None)
y_val = pd.read_csv("../CNN-2/csvTestLabel 3360x1.csv",header=None)
#x_train = np.array(x_train).reshape(-1, 28, 28, 1)
x_train = x_train.values.astype('float32')
y_train = y_train.values.astype('int32')-1

#testing images
x_val = x_val.values.astype('float32')
#testing labels
y_val = y_val.values.astype('int32')-1


x_train = x_train.reshape(-1, 32, 32, 1)
x_val = x_val.reshape(-1, 32, 32, 1)
x_train = x_train.astype("float32")/255.
x_val = x_val.astype("float32")/255.
print(x_train)
#X = X/255.0



y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

print("Constructing the Convet....")

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',
                 input_shape = (32, 32, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(28, activation='softmax'))

datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)
model.compile(loss='categorical_crossentropy', optimizer = 'Adam', metrics=["accuracy"])
#model.compile(loss='binary_crossentropy',optimizer='Adam(lr=1e-4)',metrics=['accuracy'])
print("Training the convet....")
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=20, #Increase this when not on Kaggle kernel
                           verbose=2,  #1 for ETA, 0 for silent
                           validation_data=(x_val[:400,:], y_val[:400,:]))
#y = to_categorical(y,28)
#print(y)
#model.fit(X, y,batch_size=25,epochs=100, validation_split=0.2)
"""model.fit(X, y,
          batch_size=16,
          epochs=50,
          verbose=1,
          validation_split=0.2)"""
final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
x_test = pd.read_csv("../CNN-2/csvTestImages 3360x1024.csv",header=None)
x_test = x_test.values.astype('float32')
x_test = x_test.reshape(-1, 32, 32, 1)
x_test = x_test.astype("float32")/255.
# im= cv2.imread('./C23/3094.png',cv2.IMREAD_GRAYSCALE)
# im=cv2.resize(im, (32, 32))
# im.reshape(-1, 32, 32, 1)
# img= np.array(im)
# image = np.expand_dims(img,axis=0)
# image = np.expand_dims(image,axis=3)
# image = image.astype("float32")
# print(image)
model.save('arabic_classifier_model.h5')
model.save_weights('arabicweights.h5')

#prediction = model.predict(x_test)
#pred=np.array(prediction)
#print(pred)
#final=[]
#for i in pred:
 #   final.append(i)
#print(final)
#k=0
#for i in final:
 #   if i == 1.0:
  #      ans=CATEGORIES[int(k)]
   #     break
    #else:
     #  k=k+1
#print("The inputted image classifies to "+ans)

