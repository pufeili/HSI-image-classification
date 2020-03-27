
import numpy as np
# import scipy
# import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
K.common.set_image_dim_ordering('th')
from keras.utils import np_utils
#from sklearn.cross_validation import StratifiedKFold

numPCAcomponents = 30
windowSize = 5
testRatio = 0.5

X_train = np.load("X_trainPatches_"+str(windowSize)+"PCA"
                  +str(numPCAcomponents)+"testRatio"+str(testRatio)+".npy")

y_train = np.load("y_trainPatches_"+str(windowSize)+"PCA"
                  +str(numPCAcomponents)+"testRatio"+str(testRatio)+".npy")

X_train = np.reshape(X_train,
                     (X_train.shape[0],X_train.shape[3],X_train.shape[1],X_train.shape[2]))

# convert class labels to on-hot encoding
y_train = np_utils.to_categorical(y_train)

input_shape= X_train[0].shape
print(input_shape)  #30,5,5


# number of filters
C1 = 3*numPCAcomponents

# Define the model
model = Sequential()
model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(3*C1, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(6*numPCAcomponents, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='softmax'))

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=2)

model.save('my_model'+str(windowSize)+'PCA'
           +str(numPCAcomponents)+"testRatio"+str(testRatio)+'.h5')







