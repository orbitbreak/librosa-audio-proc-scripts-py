# beat tracking example
import librosa

#filename = librosa.example('nutcracker') # bundled audio file
filename = 'data/ladymadon8bar.mp3'
print('Running beat analysis on Filename=' + filename)

waveform, samplerate = librosa.load(filename)

tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=samplerate)

print('Estimated tempo: {} bpm'.format(tempo))

# convert frame indices of beat events to timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=samplerate)
print('Beat timestamps ({} total):'.format(len(beat_times)))
for i in range(len(beat_times)):
    print('beat#{:02}  {:.3f}s'.format( i, beat_times[i] ))
