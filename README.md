
# Voice Biometrics Authentication <br>

Voice Biometrics Authentication using GMM
___

## What we need?

**Python 3.X** <br>
PyAudio<br>
GaussianMixture<br>
Scikit-Learn<br>
IPython<br>
Numpy

___


## How it works?
___

 **Import the dependencies**

 ```python

import numpy as np
import os
import glob
import pickle
import csv
import time
import pandas as pd

import pyaudio
from IPython.display import Audio, display, clear_output
import wave
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture 
import warnings
warnings.filterwarnings("ignore")

from sklearn import preprocessing
import python_speech_features as mfcc

#Create or import the list of users
users = []
if os.path.exists('./user_database/userLists.csv'):
    user_database = pd.read_csv('./user_database/userLists.csv', encoding='latin-1')
    users = user_database.values.tolist()
#print(users)
 ```
 ___

**MFCC features and Extract delta of the feature vector**

```python
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
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01, 20, appendEnergy = True, nfft=1103)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)

    #combining both mfcc features and delta
    combined = np.hstack((mfcc_feat,delta)) 
    return combined

#save the username in csv file
def userLists_CSV(lists):
    with open('./user_database/userLists.csv', 'w', newline='') as file:
        header = ['Username']
        film_writer = csv.writer(file)
        film_writer.writerow(header)
        film_writer.writerows(lists)
    file.close
```
<br>Converting audio into MFCC features and scaling it to reduce complexity of the model. Than Extract the delta of the given feature vector matrix and combine both mfcc and extracted delta to provide it to the gmm model as input.
___
  
**Adding a New User's voice**

```python
def add_user():
    
    name = input("Enter Name:")
     
    #Voice authentication
    FORMAT = pyaudio.paInt16
    CHANNELS = 2 # Mono or Stereo of input
    RATE = 44100 # Can use 44100 or 48000 Frequency depends on each device
    CHUNK = 1024 # Chunk size file can reach up to 2048
    RECORD_SECONDS = 3 # Maximum time for records
    
    source = "./voice_database/" + name
    
   
    os.mkdir(source)

    for i in range(3):
        audio = pyaudio.PyAudio()

        if i == 0:
            j = 3
            while j>=0:
                time.sleep(1.0)
                print("Speak your full name in {} seconds".format(j))
                clear_output(wait=True)

                j-=1

        elif i ==1:
            print("Speak your full name one more time")
            time.sleep(0.5)

        else:
            print("Speak your full name one last time")
            time.sleep(0.5)

        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("Recording...")
        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # saving wav file of speaker
        waveFile = wave.open(source + '/' + str((i+1)) + '.wav', 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        print("Done")

    dest =  "./gmm_models/"
    count = 1

    for path in os.listdir(source):
        path = os.path.join(source, path)

        features = np.array([])
        
        # reading audio files of speaker
        (sr, audio) = read(path)
        
        # extract 40 dimensional MFCC & delta MFCC features
        vector   = extract_features(audio,sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))
            
        # when features of 3 files of speaker are concatenated, then do model training
        if count == 3:    
            gmm = GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag', n_init = 3)
            gmm.fit(features)

            # saving the trained gaussian model
            pickle.dump(gmm, open(dest + name + '.gmm', 'wb'))
            users.append([name])
            userLists_CSV(users)
            
            print(name + ' added successfully') 
            features = np.asarray(())
            count = 0
        count = count + 1

add_user()
```
 <br>*The second part of the function *add_user()* is used to add a new user's voice into the database.*
 
 The User have to speak his/her name one time at a time as the system asks the user to speak the name for three times.
 It saves three voice samples of the user as a *wav* file. 
 
 The function *extract_features(audio, sr)* extracts 40 dimensional **MFCC** and delta MFCC features as a vector and        concatenates all  the three voice samples as features and passes it to the **GMM** model and saves user's voice model as *.gmm* file.
___
 
**Check User**

```python
# checks a registered user from database
def check_user():
    name = input("Enter name of the user:")
    name_search = [name]

    if any(list == name_search for list in users):
        print('User ' + name + ' exists')
    else:
        print('No such user !!')

check_user()
```
___

**Delete User**

```python
# deletes a registered user from database
def userLists_CSV(users):
    # Assuming this function updates the CSV file with the current user list
    pass

def delete_user(users):
    name = input("Enter name of the user:")

    # Find the user entry by name
    user_entry = next((user for user in users if user[0] == name), None)
    
    if user_entry:
        # Remove the speaker wav files and GMM model
        for file_path in glob.glob(f'./voice_database/{name}/*'):
            os.remove(file_path)
        
        os.removedirs(f'./voice_database/{name}')
        os.remove(f'./gmm_models/{name}.gmm')
        
        # Remove the user from the users list
        users.remove(user_entry)
        
        # Update the CSV file
        userLists_CSV(users)
        
        print(f'User {name} deleted successfully')
    else:
```
___

 **Voice Authentication**

 ```python
def recognize():
    # Voice Authentication
    FORMAT = pyaudio.paInt16
   CHANNELS = 2 # Mono or Stereo of input
    RATE = 44100 # Can use 44100 or 48000 Frequency depends on each device
    CHUNK = 1024 # Chunk size file can reach up to 2048
    RECORD_SECONDS = 3 # Maximum Time for recognize
    FILENAME = "./test.wav"

    audio = pyaudio.PyAudio()
   
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording")


    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # saving wav file 
    waveFile = wave.open(FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    modelpath = "./gmm_models/"

    gmm_files = [os.path.join(modelpath,fname) for fname in 
                os.listdir(modelpath) if fname.endswith('.gmm')]

    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]

    speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
                in gmm_files]
  
    if len(models) == 0:
        print("No Users in the Database!")
        return
        
    #read test file
    sr,audio = read(FILENAME)
    
    # extract mfcc features
    vector = extract_features(audio,sr)
    log_likelihood = np.zeros(len(models)) 

# Assuming your models, vector, speakers, and log_likelihood are defined

    for i in range(len(models)):
        gmm = models[i]

    # Check if score calculation results in any non-finite values (e.g., NaN)
        scores = np.array(gmm.score(vector))
        if np.isnan(scores.any()):
             print("Warning: Encountered non-finite score values. User identification might be unreliable.")
             continue  # Skip to the next model

        log_likelihood[i] = scores.sum()

# Check if any model had a positive log-likelihood (indicating a match)
    has_match = any(value > 0 for value in log_likelihood)

    if not has_match:
        print("No user found. The provided vector doesn't match any model in the set.")
    else:
        pred = np.argmax(log_likelihood)
        identity = speakers[pred]
        print("Recognized as - ", identity)  
    
recognize()
 ```
 <br> *This part of the function recognizes voice of the user as the user have to speak his/her name as the system asks.*
 
  As the user speaks his/her name the function saves the voice sample as a test.wav file and than Reads it to extract 40 dim MFCC features.
  
  Load all the pre-trained gmm models and passes the new extracted MFCC vector into the gmm.score(vector) function checking with each model one-by-one and sums the scores to calculate log_likelihood of each model. Takes the argmax value from the log_likelihood which provides the prediction of the user with highest prob distribution.
  ___
