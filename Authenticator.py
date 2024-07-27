import numpy as np
import os
import glob
import pickle
import csv
import time
import pandas as pd
# import SpeechRecognition as sr1

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
    #reading a csv file into a dataframe
    user_database = pd.read_csv('./user_database/userLists.csv', encoding='latin-1')
    #converting the dataframe to list
    users = user_database.values.tolist()
# print(users)






def add_user():
    
    name = input("Enter Name:")

     
    #Voice authentication
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    
    source = "./voice_database/" + name
    
   # It is used to create new directory at specified path
    os.mkdir(source)

    for i in range(3):
        audio = pyaudio.PyAudio()

        if i == 0:
            j = 3
            while j>=0:
                time.sleep(1.0)
                print("Speak your name in {} seconds".format(j))
                clear_output(wait=True)

                j-=1

        elif i ==1:
            print("Speak your name one more time")
            time.sleep(0.5)

        else:
            print("Speak your name one last time")
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

# checks a registered user from database
def check_user():
    name = input("Enter name of the user:")

    if any(list[0] == name for list in users):
        print('User ' + name + ' exists')
    else:
        print('No such user !!')

#convert audio to mfcc features
def extract_features(audio,rate):    
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01, 20, appendEnergy = True, nfft=1103)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)

    #combining both mfcc features and delta
    combined = np.hstack((mfcc_feat,delta)) 
    return combined

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

#save the username in csv file
def userLists_CSV(lists):
    with open('./user_database/userLists.csv', 'w', newline='') as file:
        header = ['Username']
        film_writer = csv.writer(file)
        film_writer.writerow(header)
        film_writer.writerows(lists)
    file.close


import os
import glob

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
        print('No such user !!')

# Example usage
# users = [['john', 'some_other_data'], ['jane', 'some_other_data']]
# delete_user(users)

        
       


# # deletes a registered user from database
# def delete_user():
#     name = input("Enter name of the user:")

#     if any(list[0] == name for list in users):
#         # remove the speaker wav files and gmm model
#         [os.remove(path) for path in glob.glob('./voice_database/' + name + '/*')]
#         os.removedirs('./voice_database/' + name)
#         os.remove('./gmm_models/' + name + '.gmm')
#         users.remove(list)
#         userLists_CSV(users)
        
#         print('User ' + name + ' deleted successfully')
#     else:
#         print('No such user !!')

def recognize():
    # Voice Authentication
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
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


 #**************************NEW CODE***************************************  

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


 #**************************NEW CODE***************************************


    #checking with each model one by one
    # for i in range(len(models)):
    #     gmm = models[i]         
    #     scores = np.array(gmm.score(vector))
    #     print(scores)
    #     log_likelihood[i] = scores.sum()

    # pred = np.argmax(log_likelihood)
    # identity = speakers[pred]
    
    # print("Recognized as - ", identity)
    # print(speakers)
    # print(log_likelihood)
    
#main
print('Welcome to our Voice Authunticator')
print('Select an appropriate option')
print('  1.Add user \n 2.Delete user \n 3.Check the existing user \n 4.Authenticate yourself')
val=int(input("enter the option    "))
match val :
    case 1 :
        add_user()
    case 2 :
        delete_user(users)
    case 3 :
        check_user()
    case 4 :
        recognize()