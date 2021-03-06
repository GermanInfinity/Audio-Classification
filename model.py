import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle#in python let you store binary files, sttore data
from keras.callbacks import ModelCheckpoint#saving models from keras to load up later
from cfg import Config#contain config class 


def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
        
    else:
        return None
    
    
#Generates data to be served up to the model 
def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    
    X = [] # to be converted to numpy array ;ater
    y = []
    #in order to figure out scalings for normalizing
    _min, _max = float('inf'), -float('inf')
    #We want to normalize input btw 0 and 1 for NN's
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.label==rand_class].index)
        rate, wav = wavfile.read('clean/'+file)
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate, numcep=config.nfeat, 
                                    nfilt=config.nfilt, nfft=config.nfft)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        #might check here to understand shape of data
        X.append(X_sample)
        y.append(classes.index(label))
        #Encoding names of instruments as index for NN
    config.min = _min
    config.max = _max
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=10)# need to onehotencode linear variables of Y 
    #Store in a tuple 
    config.data = (X, y)
    
    #Save in a pickle
    with open(config.p_path, 'wb') as handle:
        pickle.dump (config, handle, protocol=2)
    return X, y
        

def get_conv_model():
    model = Sequential() #modular model that adds layers 
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), 
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    #The more conv layers the opportunities to learn about features
    #Number of features to power of 2 at each layer 
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1),
                     padding='same'))
    #Done doing convolution
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5)) 
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax')) #Categorical cross entropy
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model


def get_recurrent_model(): #Meant to model features that change over time
    #shape of data for RNN is (n, time, feat)
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model
#Looking at row of pixels for something to remember to carry on to next row of 
#pixels. Scan picture try to analyse based on what you're previously seen.
    
        
# Creating pychart for class distribution 
df = pd.read_csv('numbers.csv')
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('clean/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()



#Now--start handling class imbalance when we train NN
# Audio is sampled frequently, we have to create arbitrary length of sample time
# we would use a tenth of a second. Extrememly quick to discern between 
# classifications.

n_samples = 2 * int(df['length'].sum()/0.1) # Taking double samples of data
prob_dist = class_dist / class_dist.sum()
#Random sampling of classes
choices = np.random.choice(class_dist.index, p=prob_dist)

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()

config = Config(mode='time')

if config.mode == 'conv':
    X, y = build_rand_feat()
    #mapping back to classes from one hot encoding
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()
    
elif config.mode == 'time':
    X, y = build_rand_feat() # Function to build data so it'll be preprocessed 
                             # to be pushed through models
    #mapping back to classes from one hot encoding
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()
    print ("Here")
    
class_weight = compute_class_weight('balanced',
                                    np.unique(y_flat),
                                    y_flat)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1,
                             mode='max', save_best_only=True, save_weights_only=False,
                             period=1)
#When you save models in keras, it looks for a weight file. Now store 
#model architecutre and weights in one .model file

model.fit(X, y, epochs=10, batch_size=32,
          shuffle=True, class_weight=class_weight, validation_split=0.1, 
          callbacks=[checkpoint])
#validation split take bottom 10% to validate it, must shuffle data before 
#validating. 
model.save(config.model_path)