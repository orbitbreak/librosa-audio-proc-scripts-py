# autotune algorithms
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import scipy.signal as sig
from functools import partial
import psola
import argparse

SEMITONES_IN_OCTAVE = 12
INPUT_FILE = "inputfile.wav"
OUTPUT_FILE = "outputfile.wav"

def closest_pitch(freq):
    NaNs  = np.isnan(freq)
    midiNote = np.around(librosa.hz_to_midi(freq))
    midiNote[NaNs ] = np.nan
    return librosa.midi_to_hz(midiNote)
def closest_corrected_pitch(freq, scale_oct):
    if np.isnan(freq):
        return np.nan
    degrees=degrees_from(scale_oct)
    midiNote = librosa.hz_to_midi(freq) # !!!
    degree = midiNote % SEMITONES_IN_OCTAVE
    degree_id = np.argmin( np.abs( degrees - degree ) )
    degree_difference = degree - degrees[degree_id]
    midiNote -= degree_difference
    return librosa.midi_to_hz( midiNote)

def degrees_from(scale_oct: str):
    degrees = librosa.key_to_degrees(scale_oct)
    degrees = np.concatenate( (degrees,[degrees[0] + SEMITONES_IN_OCTAVE] ) )
    return degrees

def aclosest_corrected_pitch(freq, scale_oct):
    corrected_pitch = np.zeros_like(freq)
    for i in np.arange( freq.shape[0] ):
        corrected_pitch[i] = closest_corrected_pitch( freq[i], scale_oct )
    smoothed_corrected_pitch = sig.medfilt(corrected_pitch, kernel_size=11 )
    smoothed_corrected_pitch[ np.isnan(smoothed_corrected_pitch) ] = freq[ np.isnan(smoothed_corrected_pitch)]
    return smoothed_corrected_pitch

def autotune(audio, sr, correction_function, plot=False):
    frame_length = 2048
    hop_length = frame_length # 4
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')
    f0, voiced_flag, voiced_probabilities = librosa.pyin( audio, frame_length=frame_length, hop_length=hop_length, sr=sr, fmin=fmin, fmax=fmax)
    corrected_f0 = correction_function(f0)
    if plot:
        time_points = librosa.times_like(f0)
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        log_stft = librosa.amplitude_to_db( np.abs(stft), ref=np.max )
        fig, ax = plt.subplots()
        img = librosa.display.specshow( log_stft, x_axis='time', y_axis='log', ax=ax)
        fig.colorbar( img, ax=ax, format="%+2.f db")
        ax.plot(time_points, f0, label='orig pitch', color='green', linewidth=2)
        ax.plot(time_points, corrected_f0, label='autotuned pitch', color='red', linewidth=2)
        ax.legend(loc='upper right')
        plt.savefig('autotune_output.png', dpi=300, bbox_inches='tight')
    return psola.vocode(audio, sample_rate=int(sr), target_pitch=corrected_f0, fmin=fmin, fmax=fmax)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--plot', action='store_true', default=False)
    argparser.add_argument('--algo', choices=['closest','scale'], default='closest')
    argparser.add_argument('--scale', type=str, help='uses librosa.key_to_degrees for scale correction method only (non-default)')
    args = argparser.parse_args()

    y, sr = librosa.load( INPUT_FILE, sr=None, mono=False)
    if y.ndim > 1:
        y = y[0,:]
    correction_function = closest_pitch if args.algo=='closest' else partial(aclosest_corrected_pitch, scale=args.scale)

    pitch_corrected_y = autotune(y, sr, correction_function, args.plot)
    sf.write( OUTPUT_FILE, pitch_corrected_y, sr)

if __name__=='__main__':
    main()

