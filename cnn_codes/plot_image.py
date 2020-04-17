import os
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
import spectral
from keras.models import load_model


'''
hyper param
'''
numPCAcomponents = 30
windowSize = 5
testRatio = 0.9


def loadSalinasData():
    data_path = os.path.join('./datasets')
    data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
    labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    return data, labels

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def Patch(data,height_index,width_index):
    #transpose_array = data.transpose((2,0,1))
    #print transpose_array.shape
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch

# load the model architecture and weights
model = load_model('./model/my_model_'+str(windowSize)+'PCA'
                   +str(numPCAcomponents)+"_testRatio_"+str(testRatio)+'.h5')

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
