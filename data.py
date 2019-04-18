import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
import my_audio_plot

df = pd.read_csv('numbers.csv') #30 .wav files for each instrument, numbers
df.set_index('fname', inplace=True)

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

#Noise threshold detection
# Calculate envelope of the signal--threshold of desired magnitude
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    #aggregate mean of signals in window
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean: 
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask
    
#Reading .wav files and computing the 
#length: ./wav_file_signal / ./wav_file_rate
#Added length of .wav file as column to dataframe
for f in df.index:
    rate, signal = wavfile.read('numWavfiles/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate #returns length of .wav
#Appends length 

#Make list of class -- instruments
classes = list(np.unique(df.label))
#Make list of instruments with mean of each instruments length
class_dist = df.groupby(['label'])['length'].mean() 

#Plot mean distribution for all classes
fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08) #y variable moves title upward 
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal') #Makes chart look like a circle instead of an ellipse
plt.show()
df.reset_index(inplace=True) #moves filename back into its own column

#Data presents a lot of class inbalance. 
#We can acheive class balance later on when we model.

#Dictionary of properties we care about for each class
signals = {}
fft = {}
fbank = {}
mfccs = {}
sig = my_audio_plot.my_audio_plot()
#Visualization of time-series
    #Access .wav file in each class
for c in classes: 
    wav_file = df[df.label == c].iloc[0, 0]
    signal, rate = librosa.load('numWavfiles/'+wav_file, sr=44100)

    mask = envelope(signal, rate, 0.0005)
    signal = signal[mask]
    
    signals[c] = signal 
    fft[c] = calc_fft(signal, rate)
    
    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103) #nfft: how many 
   #points alloted to calculate fft  sampling freqeuncy(44.1) divded by 40. 
    fbank[c] = bank
    mel = mfcc(signal[:rate], rate, numcep=26, nfft=1103).T#numcep Num of ceptuals ketp after DCT
    mfccs[c] = mel
    
sig.plot_signals(signals)
plt.show()

sig.plot_fft(fft)
plt.show()

sig.plot_fbank(fbank)
plt.show()

sig.plot_mfccs(mfccs)
plt.show()

# We notice empty low magnitude portions in our signals. 
# We want to get rid of these so our algorithms take good features. 
# Process is called noise threshold detection


#Downsample audio -- put mask over audio
# making audio clean for modelling
#Generate clean audio files ready for modelling 
if len(os.listdir('clean')) == 0:
    for f in tqdm(df.fname):
        signal, rate = librosa.load('numWavfiles/'+f, sr=16000)
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='clean/'+f, rate=rate, data=signal[mask])