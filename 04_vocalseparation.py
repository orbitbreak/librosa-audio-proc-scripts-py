# vocal separation example
import librosa
import matplotlib.pyplot as plt
import numpy as np

#filename = librosa.example('nutcracker') # bundled audio file
filename = 'data/ladymadon8bar.mp3'

print('Generating voice-separation spectrogram from Filename=' + filename)
waveform, samplerate = librosa.load(filename)

# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(waveform))

# 5 second slice spectrum, wiggly lines are vocals
idx = slice(*librosa.time_to_frames([1, 6], sr=samplerate))
#fig, ax = plt.subplots()
#img = librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
#                         y_axis='log', x_axis='time', sr=samplerate, ax=ax)
#fig.colorbar(img, ax=ax)
#plt.show() # show 5 second slice spectrum, test before separation logic

# compare frames with cosine similarity, agg similar frames by per-freq median
# ...require that similar frames separated by at least 1 second
S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, 
                                        metric='cosine',
                                        width=int(librosa.time_to_frames(2, sr=samplerate)))
# output must be smaller than input, take pointwise minimum with input spectrum
S_filter = np.minimum(S_full, S_filter)

# margin to reduce bleed, diff for foreground/background separation
margin_i, margin_v = 2, 10
power = 2

mask_i = librosa.util.softmask(S_filter, margin_i*(S_full - S_filter),
                                power=power)
mask_v = librosa.util.softmask(S_full - S_filter,
                                margin_v * S_filter,
                                power=power)

# multiply masks with input spectrum to separate 
S_foreground = mask_v * S_full
S_background = mask_i * S_full

# plot same slice, but separated
fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
img = librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                                y_axis='log', x_axis='time', sr=samplerate, ax=ax[0])
ax[0].set(title='Full spectrum')
ax[0].label_outer()

librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                                y_axis='log', x_axis='time', sr=samplerate, ax=ax[1])
ax[1].set(title='Background (isntruments)')
ax[1].label_outer()

librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                                y_axis='log', x_axis='time', sr=samplerate, ax=ax[2])
ax[2].set(title='Foreground (vocals)')

fig.colorbar(img, ax=ax)
plt.show() # show spectrogram of final separated results

#separated_vocals = librosa.istft(S_foreground * phase)


