import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import random
from random import shuffle
from skimage.transform import rotate
import scipy.ndimage


def loadSalinasData():
    data_path = os.path.join('./datasets')
    data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
    labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    return data, labels

def splitTrainTestSet(X, y, testRatio=0.10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio,
                                                        random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def oversampleWeakClasses(X, y):
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts
    # repeat for every label and concat
    newX = X[y == uniqueLabels[0], :, :, :].repeat(round(labelInverseRatios[0]), axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y== label,:,:,:].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :, :, :]
    newY = newY[rand_perm]
    return newX, newY

def standartizeData(X):
    newX = np.reshape(X, (-1, X.shape[2]))
    scaler = preprocessing.StandardScaler().fit(newX)
    newX = scaler.transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
    return newX, scaler

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createPatches(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def AugmentData(X_train):
    for i in range(int(X_train.shape[0] / 2)):
        patch = X_train[i, :, :, :]
        num = random.randint(0, 2)
        if (num == 0):
            flipped_patch = np.flipud(patch)
        if (num == 1):
            flipped_patch = np.fliplr(patch)
        if (num == 2):
            no = random.randrange(-180, 180, 30)
            flipped_patch = scipy.ndimage.interpolation.rotate(patch, no, axes=(1, 0),
                                                               reshape=False, output=None, order=3, mode='constant',
                                                               cval=0.0, prefilter=False)
    patch2 = flipped_patch
    X_train[i, :, :, :] = patch2

    return X_train

def savePreprocessedData(X_trainPatches, X_testPatches, y_trainPatches, y_testPatches, windowSize, wasPCAapplied = False, numPCAComponents = 0, testRatio = 0.25):
    if wasPCAapplied:
        with open("X_trainPatches_" + str(windowSize) + "PCA" + str(numPCAComponents) + "testRatio" + str(testRatio) + ".npy", 'bw') as outfile:
            np.save(outfile, X_trainPatches)
        with open("X_testPatches_" + str(windowSize) + "PCA" + str(numPCAComponents) + "testRatio" + str(testRatio) + ".npy", 'bw') as outfile:
            np.save(outfile, X_testPatches)
        with open("y_trainPatches_" + str(windowSize) + "PCA" + str(numPCAComponents) + "testRatio" + str(testRatio) + ".npy", 'bw') as outfile:
            np.save(outfile, y_trainPatches)
        with open("y_testPatches_" + str(windowSize) + "PCA" + str(numPCAComponents) + "testRatio" + str(testRatio) + ".npy", 'bw') as outfile:
            np.save(outfile, y_testPatches)
    else:
        with open("../preprocessedData/XtrainWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
            np.save(outfile, X_trainPatches)
        with open("../preprocessedData/XtestWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
            np.save(outfile, X_testPatches)
        with open("../preprocessedData/ytrainWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
            np.save(outfile, y_trainPatches)
        with open("../preprocessedData/ytestWindowSize" + str(windowSize) + ".npy", 'bw') as outfile:
            np.save(outfile, y_testPatches)

'''
myFile = open('global_variables.txt', 'r')
file = myFile.readlines()[:]

for line in file:
    if line[0:3] == "win":
        ds = line.find('=')
        windowSize = int(line[ds+1:-1],10)

    elif line[0:3] == "num":
        ds = line.find('=')
        numPCAcomponents = int(line[ds+2:-1],10)

    else:
        ds = line.find('=')
        testRatio = float(line[ds+1:])
'''

numPCAcomponents = 30
windowSize = 5
testRatio = 0.5

Dataset, GroundTruth = loadSalinasData()
Dataset,pca = applyPCA(Dataset,numPCAcomponents)
XPatches, yPatches = createPatches(Dataset, GroundTruth, windowSize=windowSize)
X_train, X_test, y_train, y_test = splitTrainTestSet(XPatches, yPatches, testRatio)
X_train, y_train = oversampleWeakClasses(X_train, y_train)
X_train = AugmentData(X_train)
savePreprocessedData(X_train, X_test, y_train, y_test,
                     windowSize = windowSize,wasPCAapplied=True,
                     numPCAComponents = numPCAcomponents,testRatio = testRatio)



import numpy as np
import scipy
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
K.common.set_image_dim_ordering('th')
from keras.utils import np_utils
#from sklearn.cross_validation import StratifiedKFold

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

model.fit(X_train, y_train, batch_size=32, epochs=5)

import h5py
from keras.models import load_model

model.save('my_model'+str(windowSize)+'PCA'
           +str(numPCAcomponents)+"testRatio"+str(testRatio)+'.h5')

# Import the necessary libraries
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import itertools
import spectral
import matplotlib
# %matplotlib inline

def reports(X_test, y_test):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow'
        , 'Fallow_smooth', 'Stubble', 'Celery',
                    'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
                    'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 'Vinyard_untrained',
                    'Vinyard_vertical_trellis']

    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    avgAcc = []
    classArray = []
    for c in range(len(confusion)):
        recallSoc = confusion[c][c] / sum(confusion[c])
        classArray += [recallSoc]
    avgAcc.append(sum(classArray) / len(classArray))
    avg_accuracy = np.mean(avgAcc)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    overall_loss, overall_accu = model.evaluate(X_test, y_test, verbose=False)

    return classification, confusion, kappa, avg_accuracy, overall_accu

def Patch(data,height_index,width_index):
    #transpose_array = data.transpose((2,0,1))
    #print transpose_array.shape
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch

X_test  = np.reshape(X_test, (X_test.shape[0],
                              X_test.shape[3], X_test.shape[1], X_test.shape[2]))
y_test = np_utils.to_categorical(y_test)

# load the model architecture and weights
model = load_model('my_model'+str(windowSize)+'PCA'
                   +str(numPCAcomponents)+"testRatio"+str(testRatio)+'.h5')

# Using the pretrained model make predictions and print the results into a report
classification, confusion,  kappa, overall_accuracy, Average_accuracy = reports(X_test,y_test)
print('Classification_report:\n')
print('{}\n'.format(classification))
print('Confusion_matrix :\n')
print('{}\n'.format(confusion))
print('Kappa value : {}\n'.format(kappa))
print('Overall accuracy : {}\n'.format(overall_accuracy))
print('Average_accuracy : {}\n'.format(Average_accuracy))

# load the original image
Dataset, GroundTruth = loadSalinasData()

Dataset,pca = applyPCA(Dataset,numPCAcomponents)

height = GroundTruth.shape[0]
width = GroundTruth.shape[1]
PATCH_SIZE = windowSize
numPCAcomponents = numPCAcomponents

# calculate the predicted image
outputs = np.zeros((height,width))

for i in range(height-PATCH_SIZE+1):
    for j in range(width-PATCH_SIZE+1):
        target = GroundTruth[int(i+PATCH_SIZE/2), int(j+PATCH_SIZE/2)]
        if target == 0 :
            continue
        else :
            image_patch=Patch(Dataset,i,j)
            #print (image_patch.shape)
            X_test_image = image_patch.reshape(1,image_patch.shape[2],
                                               image_patch.shape[0],image_patch.shape[1]).astype('float32')
            prediction = (model.predict_classes(X_test_image))
            outputs[int(i+PATCH_SIZE/2)][int(j+PATCH_SIZE/2)] = prediction+1

# Plot the Ground Truth Image
ground_truth = spectral.imshow(classes = GroundTruth,figsize =(5,5))
# Plot the Predicted image
predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(5,5))

from tqdm import tqdm
import pandas as pd
def extract_pixels(dataset, ground_truth):
    df = pd.DataFrame()
    for i in tqdm(range(dataset.shape[2])):
        df = pd.concat([df, pd.DataFrame(dataset[:, :, i].ravel())], axis=1)
    df = pd.concat([df, pd.DataFrame(ground_truth.ravel())], axis=1)
    df.columns = [f'band-{i}' for i in range(1, 1+dataset.shape[2])]+['class']
    return df

df = extract_pixels(Dataset,GroundTruth)

df.to_csv('Dataset.csv', index=False)
df = pd.read_csv('Dataset.csv')
df.head()
df.tail()

Dataset = df.iloc[:, :-1].values
GroundTruth = df.iloc[:, -1].values
# Dataset.shape, GroundTruth.shape      ((111104, 30), (111104,))

from matplotlib import pyplot as plt

def plot_signature(df):
    plt.figure(figsize=(12, 6))
    pixel_no = np.random.randint(df.shape[0])
    plt.plot(range(1, 31), df.iloc[pixel_no, :-1].values.tolist(), 'b--', label= f'Class - {df.iloc[pixel_no, -1]}')
    plt.legend()
    plt.title(f'Pixel({pixel_no}) signature', fontsize=14)
    plt.xlabel('Band Number', fontsize=14)
    plt.ylabel('Pixel Intensity', fontsize=14)
    plt.show()

plot_signature(df)

print(f"Unique Class Labels: {df.loc[:, 'class'].unique()}")
#Unique Class Labels: [ 0  6  7  4  5 15  8  3  2  1 11 12 13 14 10  9 16]
df.loc[:, 'class'].value_counts()
'''
0     56975
8     11271 
15     7268
9      6203
6      3959
2      3726
7      3579
10     3278
5      2678
1      2009
3      1976
12     1927
16     1807
4      1394
14     1070
11     1068
13      916
Name: class, dtype: int64
'''
df[df['class']== 5][0:2000]


