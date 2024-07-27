import tensorflow as tf
import numpy as np
import os
import sys
import pickle
from numpy import genfromtxt
from keras import backend as K
from keras.models import load_model
# import dependencies for voice biometrics
import pyaudio
from IPython.display import Audio, display, clear_output
import wave
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture 
import warnings
warnings.filterwarnings("ignore")

from sklearn import preprocessing
import python_speech_features as mfcc

K.set_image_data_format('channels_first')
np.set_printoptions(threshold=sys.maxsize)

#Calculate and returns the delta of given feature vector matrix
def calculate_delta(array):
    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

#convert audio to mfcc features
def extract_features(audio,rate):    
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,20,appendEnergy = True, nfft=1103)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)

    #combining both mfcc features and delta
    combined = np.hstack((mfcc_feat,delta)) 
    return combined
