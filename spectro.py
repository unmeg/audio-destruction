import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--wav1', type=str, help='file path for audio 1')
parser.add_argument('--wav2', type=str, help='file path for audio 2')
options = parser.parse_args()

audz1 = options.wav1
sample_rate1, samples1 = wavfile.read(audz1)
samples1 = samples1.astype(np.float)

audz2 = options.wav2
sample_rate2, samples2 = wavfile.read(audz2)
samples2 = samples2.astype(np.float)

savename = options.wav1 +'_' + options.wav2

def get_spec(x, n_fft=2048):
    S = librosa.stft(x, n_fft)
    S = librosa.amplitude_to_db(librosa.magphase(S)[0], ref=np.max)
    return S


def plot_spec(S, fs, title='Spectrogram'):
    librosa.display.specshow(S, sr=fs, y_axis='linear', x_axis='time')
    plt.title(title)

def spectrum_comparison(audio_one, audio_one_sr, audio_two, audio_two_sr, save_file=None):
    audio_one_spec = get_spec(audio_one)
    audio_two_spec = get_spec(audio_two) 

    plt.subplot(1, 2, 1)
    title1 = str('Signal 1: ' + options.wav1)
    plot_spec(audio_one_spec, audio_one_sr, title1)

    plt.subplot(1, 2, 2)
    title2 = str('Signal 2: ' + options.wav2)
    plot_spec(audio_two_spec, audio_two_sr, title2)

    if save_file is not None:
        plt.savefig(save_file + '_spec.png')

    plt.show()

    if audio_one_sr == audio_two_sr:
        f, pxy = signal.csd(audio_one, audio_two, audio_one_sr) # gives us correlation between 1 and 2
        plt.semilogy(f, np.abs(pxy))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('CSD [V**2/Hz]')
        
        if save_file is not None:
            plt.savefig(save_file + '_csd.png')

        plt.show()
    else:
        print('different sample rates')
        f, pxy = signal.csd(audio_one, audio_two)
        plt.semilogy(f, np.abs(pxy))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('CSD [V**2/Hz]')
        
        if save_file is not None:
            plt.savefig(save_file + '_csd.png')

        plt.show()


    


if __name__ == '__main__':
    spectrum_comparison(samples1, sample_rate1, samples2, sample_rate2, savename)
