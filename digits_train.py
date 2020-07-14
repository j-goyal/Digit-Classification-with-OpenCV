import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D

import pickle

# ================Parameters==========================

path = 'mydata'
testRatio = 0.2
validRatio = 0.2
imageDimensions = (32,32,3)
batchSizeVal= 50
epochsVal = 10
stepsPerEpochVal = 2000

#===========================================

images = []  # append all images of digits 0-9 in it
classNo = [] # consits of class id for each image

mylist = os.listdir(path)
noOfClasses = len(mylist)

print("total no. of classes detected", noOfClasses)

print("Importing classes......")
for x in range(noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        classNo.append(x)
    print(x,end=" ")

print()

images = np.array(images)
classNo = np.array(classNo)
# print(images.shape)

# ========================== Spliting the data ===============================
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio, random_state=0)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validRatio, random_state=0)

#print(X_train.shape, X_test.shape, X_validation.shape)

noOfSamples = []  # how many images of each class
for x in range(noOfClasses):
    noOfSamples.append(len(np.where(y_train==x)[0]))

print(noOfSamples)


def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255  # better for training model
    return img


# img = preProcessing(X_train[0])
# img = cv2.resize(img,(100,100))
# cv2.imshow("image", img)
# cv2.waitKey(0)


# Apply preProcessing function to each array
X_train = np.array(list(map(preProcessing,X_train)))  # map fun calls preProcessing for every image in X_train
X_test = np.array(list(map(preProcessing,X_test)))
X_validation = np.array(list(map(preProcessing,X_validation)))


# Add depth to image
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)


dataGen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10)

dataGen.fit(X_train)



# ONE HOT ENCODING OF MATRICES
y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)


def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1, input_shape=(imageDimensions[0],imageDimensions[1],1), activation='relu')))
    model.add((Conv2D(noOfFilters,sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))

    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))

    model.add(Dropout(0.5))
 
    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
 
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 

model = myModel()
print(model.summary())

# STARTING THE TRAINING PROCESS
history = model.fit_generator(dataGen.flow(X_train,y_train,
                                 batch_size=batchSizeVal),
                                 steps_per_epoch=stepsPerEpochVal,
                                 epochs=epochsVal,
                                 validation_data=(X_validation,y_validation),
                                 shuffle=1)

# PLOT THE RESULTS  
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
 

# EVALUATE USING TEST IMAGES
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

# SAVE THE TRAINED MODEL 
pickle_out= open("model_trained.p", "wb")
pickle.dump(model,pickle_out)
pickle_out.close()







