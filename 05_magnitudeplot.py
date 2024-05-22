# plotting magnitude of audio signal
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path

INPUT_FILE = "input_signal.wav"
COLOR = "#eff000"
SAVE_PARAMS = {"dpi":300, "bbox_inches":"tight", "transparent":True}
XTICKS = np.array([31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000])
XTICK_LABELS = np.array(["31.25", "62.5", "125", "250", "500", "1k", "2k", "4k", "8k"])
plt.rcParams.update({"font.size":20})

def save_spectrum(output_filepath):
    plt.savefig(output_filepath, **SAVE_PARAMS)

def plot_spectrum_and_save(freqs, magnitude_spec, output_filepath: Path):
    print('Top of plot_spectrum_and_save')
    plt.figure(figsize=(12,6))
    plt.plot(freqs, magnitude_spec, COLOR)
    xlim = [freqs[0], freqs[-1]]
    plt.xlim(xlim)
    plt.xlabel("freq Hz")
    plt.hlines(0, xlim[0], xlim[1], colors="k")
    plt.xticks(None,None)
    plt.yticks([])
    plt.ylabel("magnitude")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    save_spectrum(output_filepath)
    plt.close()

def plot_spectrum_db_and_save(freqs, magnitude_spec, output_filepath: Path):
    print('Top of plot_spectrum_db_and_save')
    plt.figure(figsize=(12,6))
    plt.plot(freqs, magnitude_spec, COLOR)
    xlim = [0, freqs[-1]]
    plt.xlim(xlim)
    plt.ylim([-60,0])
    plt.grid()
    plt.xlabel("freq Hz")
    plt.ylabel("magnitude")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_spectrum(output_filepath)
    plt.close()

def plot_spectrum_db_in_octaves_and_save(freqs, magnitude_spec, output_filepath: Path):
    print('Top of plot_spectrum_db_in_octaves_and_save')
    plt.figure(figsize=(12,6))
    plt.semilogx(freqs, magnitude_spec, COLOR)
    plt.ylim([-60,0])
    plt.xticks(XTICKS, XTICK_LABELS)
    min_x = 29
    plt.xlim([min_x, freqs[-1]])
    plt.grid()
    plt.xlabel("freq Hz")
    plt.ylabel("magnitude")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_spectrum(output_filepath)
    plt.close()

print('Starting script')
input_signal, sample_rate = sf.read( INPUT_FILE )
print('Read input file={}, at sample_rate={}'.format(INPUT_FILE, str(sample_rate)))
magnitude_spec = np.abs(np.fft.rfft(input_signal))
freqs = np.fft.rfftfreq(input_signal.shape[0], 1/sample_rate)

plot_spectrum_and_save( freqs, magnitude_spec, "output_magnitude_spec.png" )

normalized_magnitude_spec = magnitude_spec / np.amax(magnitude_spec)
magnitude_spec_db = librosa.amplitude_to_db( normalized_magnitude_spec )

plot_spectrum_db_and_save( freqs, magnitude_spec_db, "output_magnitude_spec_db.png" )

plot_spectrum_db_in_octaves_and_save( freqs, magnitude_spec_db, "output_magnitude_spec_db_in_octaves.png" )

print('Finished script')

