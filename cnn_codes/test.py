# Import the necessary libraries
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import itertools
import spectral
import matplotlib as plt
import scipy.io as sio
import os
from sklearn.decomposition import PCA
# %matplotlib inline

def reports(X_test, y_test):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow',
                    'Fallow_smooth','Stubble','Celery','Grapes_untrained',
                    'Soil_vinyard_develop','Corn_senesced_green_weeds','Lettuce_romaine_4wk',
                    'Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                    'Vinyard_untrained','Vinyard_vertical_trellis']

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


numPCAcomponents = 30
windowSize = 5
testRatio = 0.9

X_test = np.load("./salinas/X_testPatches_" + str(windowSize) + "PCA" +
                  str(numPCAcomponents) + "_testRatio_" + str(testRatio) + ".npy")

y_test = np.load("./salinas/y_testPatches_" + str(windowSize) + "PCA" +
                  str(numPCAcomponents) + "_testRatio_" + str(testRatio) + ".npy")


X_test  = np.reshape(X_test, (X_test.shape[0],
                              X_test.shape[3], X_test.shape[1], X_test.shape[2]))
y_test = np_utils.to_categorical(y_test)

# load the model architecture and weights
model = load_model('./model/my_model_'+str(windowSize)+'PCA'
                   +str(numPCAcomponents)+"_testRatio_"+str(testRatio)+'.h5')

# Using the pretrained model make predictions and print the results into a report
classification, confusion,  kappa, overall_accuracy, Average_accuracy = reports(X_test,y_test)
print('Classification_report:\n')
print('{}\n'.format(classification))
print('Confusion_matrix :\n')
print('{}\n'.format(confusion))
print('Kappa value : {}\n'.format(kappa))
print('Overall accuracy : {}\n'.format(overall_accuracy))
print('Average_accuracy : {}\n'.format(Average_accuracy))



