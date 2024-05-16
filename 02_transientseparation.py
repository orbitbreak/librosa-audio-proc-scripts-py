# feature extraction and transient separation
import librosa
import numpy as np

#filename = librosa.example('nutcracker') # bundled audio file
filename = 'data/ladymadon8bar.mp3'

print('Running feature extraction and harmonic/transient separation on Filename=' + filename)

waveform, samplerate = librosa.load(filename)
hop_length = 512 # loads at 22050 Hz, so 512 samples = 23ms

# separate harmonics and percussives
waveform_harmonic, waveform_percussive = librosa.effects.hpss(waveform) # tonals and transients

# beat track percussive signal
tempo, beat_frames = librosa.beat.beat_track(y=waveform_percussive, sr=samplerate)

# MFCC features from raw signal (Mel-Frequency Cepstral Coefficients)
#   numpy.ndarray of shape (n_mfcc, T), where T is track duration in frames
mfcc = librosa.feature.mfcc(y=waveform, sr=samplerate, hop_length=hop_length, n_mfcc=13)
# smoothed first order differences (delta features)
mfcc_delta = librosa.feature.delta(mfcc)

# sum and sync all beat events
#    util.sync aggs cols of input, num cols is beat_frames
#    Each col beat_mfcc_delta[:, k] is avg of input cols between beat_frames[k] and k+1
beat_mfcc_delta = librosa.util.sync( np.vstack([mfcc, mfcc_delta]), beat_frames )

# chroma features from harmonic signal
#    ndarray of shape (12, T), each row is pitch note
chromagram = librosa.feature.chroma_cqt(y=waveform_harmonic, sr=samplerate)

# aggregate chroma features of beats, using median
beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

# stack all beat-synced features
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
# feature matrix is shape (12, 13, 13, # beat intervals)
print('Beat features matrix, of shape (12, 13, 13, # beat intervals) : ')
print(beat_features)
