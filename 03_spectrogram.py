# spectrogram example
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#filename = librosa.example('nutcracker') # bundled audio file
filename = 'data/ladymadon8bar.mp3'

print('Generating spectrogram from Filename=' + filename)
waveform, samplerate = librosa.load(filename)
plt.figure(figsize=(10,4) )
D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
librosa.display.specshow(D, sr=samplerate, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()
